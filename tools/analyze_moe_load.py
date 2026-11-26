#!/usr/bin/env python3
"""Analyze TorchTurbo pre-dispatch MoE route dumps with NumPy on CPU."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np


def aggregate_load(
    counts: np.ndarray, ranks: list[int], expert_placement: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate source rows and map each expert to its recorded EP rank."""
    expert_load = counts.sum(axis=0, dtype=np.int64)
    rank_indices = {rank: index for index, rank in enumerate(ranks)}
    owners = [rank_indices[rank] for rank in expert_placement]
    rank_load = np.zeros(len(ranks), dtype=np.int64)
    np.add.at(rank_load, owners, expert_load)
    return expert_load, rank_load


def _stats(load: np.ndarray) -> dict:
    mean = float(load.mean())
    maximum = int(load.max())
    return {
        "min": int(load.min()),
        "max": maximum,
        "mean": mean,
        "max_over_mean": maximum / mean if mean else None,
    }


def _rank_stats(load: np.ndarray, ranks: list[int]) -> dict:
    stats = _stats(load)
    stats["hot_global_ranks"] = [rank for rank, value in zip(ranks, load) if value == stats["max"]]
    return stats


def _buffer_bytes(rank_load: np.ndarray, counts: np.ndarray, metadata: dict, local_experts: int) -> dict:
    row_bytes = metadata["hidden_size"] * metadata["bytes_per_element"]
    expert_output_bytes = int(rank_load.max()) * row_bytes
    return {
        "fc2_output_max_bytes": expert_output_bytes,
        "combine_permutation_max_bytes": expert_output_bytes if local_experts > 1 else 0,
        "combine_returned_packed_max_bytes": int(counts.sum(axis=1, dtype=np.int64).max()) * row_bytes,
    }


def simulate_ep(counts: np.ndarray, metadata: dict, ep_size: int) -> dict:
    """Split source rows on the same world mesh and replicate experts per subgroup."""
    ranks = metadata["ep_group_global_ranks"]
    original_ep, num_experts = counts.shape
    world_size = metadata["world_size"]
    sp_size = metadata["sp_size"]
    if (
        not 1 <= ep_size <= original_ep
        or original_ep % ep_size
        or world_size % ep_size
        or world_size % sp_size
        or num_experts % ep_size
    ):
        raise ValueError("EP must divide the recorded EP, world size and number of experts")

    _, actual_load = aggregate_load(counts, ranks, metadata["expert_placement"])
    if ep_size < original_ep and (
        ranks != list(range(ranks[0], ranks[0] + original_ep)) or ranks[0] % original_ep
    ):
        raise ValueError("EP group does not match TorchTurbo's contiguous world mesh")

    num_local_experts = num_experts // ep_size
    simulated = np.empty(original_ep, dtype=np.int64)
    subgroups = []
    for start in range(0, original_ep, ep_size):
        stop = start + ep_size
        subgroup_ranks = ranks[start:stop]
        if ep_size == original_ep:
            rank_load = actual_load
        else:
            placement = [rank for rank in subgroup_ranks for _ in range(num_local_experts)]
            _, rank_load = aggregate_load(counts[start:stop], subgroup_ranks, placement)
        simulated[start:stop] = rank_load
        subgroups.append({
            "global_ranks": subgroup_ranks,
            "rank_load": rank_load.tolist(),
            "rank_stats": _rank_stats(rank_load, subgroup_ranks),
        })

    subgroup_imbalances = (group["rank_stats"]["max_over_mean"] for group in subgroups)
    max_subgroup_imbalance = max((value for value in subgroup_imbalances if value is not None), default=None)
    return {
        "kind": "counterfactual_fixed_routing",
        "ep_size": ep_size,
        "identity": ep_size == original_ep,
        "num_local_experts": num_local_experts,
        "actual_max_rank_load": int(actual_load.max()),
        "max_rank_load_delta": int(simulated.max() - actual_load.max()),
        "rank_stats": _rank_stats(simulated, ranks),
        "max_subgroup_imbalance": max_subgroup_imbalance,
        "buffer_estimates": _buffer_bytes(simulated, counts, metadata, num_local_experts),
        "subgroups": subgroups,
    }


def compare_phases(forward: np.ndarray, recompute: np.ndarray, metadata: dict) -> dict:
    """Compare route counts and received rank loads for one logical layer call."""
    ranks = metadata["ep_group_global_ranks"]
    owners = metadata["expert_placement"]
    diff = recompute.astype(np.int64) - forward
    _, forward_load = aggregate_load(forward, ranks, owners)
    _, recompute_load = aggregate_load(recompute, ranks, owners)
    rank_delta = recompute_load - forward_load
    return {
        "num_changed_elements": int(np.count_nonzero(diff)),
        "max_abs_count_delta": int(np.abs(diff).max()),
        "global_ranks": ranks,
        "rank_load_delta": rank_delta.tolist(),
        "max_abs_rank_load_delta": int(np.abs(rank_delta).max()),
    }


def _read_metadata(manifest: Path, skipped: list[dict]) -> Iterator[dict]:
    with manifest.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                metadata = json.loads(line)
                if metadata["schema_version"] != 1 or metadata["phase"] not in ("forward", "recompute"):
                    raise ValueError("unsupported schema or phase")
                # Validate the fields that determine row ordering and pairing.
                if type(metadata["microstep"]) is not int or metadata["microstep"] < 0:
                    raise ValueError("invalid microstep")
                if type(metadata["layer_id"]) not in (int, str):
                    raise ValueError("invalid layer ID")
                _call_key(metadata)
            except (KeyError, TypeError, ValueError) as exc:
                skipped.append({"manifest": str(manifest), "line": line_number, "reason": str(exc)})
                continue
            yield metadata


def _load_matrix(directory: Path, metadata: dict) -> np.ndarray:
    path = (directory / metadata["matrix_path"]).resolve()
    if not path.is_relative_to(directory) or path.suffix != ".npy":
        raise ValueError("invalid matrix path")
    counts = np.load(path, allow_pickle=False)
    if not isinstance(counts, np.ndarray):
        counts.close()
        raise ValueError("expected a standalone NPY matrix")
    ep_size, num_experts = metadata["ep_size"], metadata["num_experts"]
    ranks = metadata["ep_group_global_ranks"]
    owners = metadata["expert_placement"]
    ownership = Counter(owners)
    if (
        counts.shape != (ep_size, num_experts)
        or counts.shape != tuple(metadata["matrix_shape"])
        or counts.size == 0
        or counts.dtype.name not in ("int32", "int64")
        or counts.dtype.name != metadata["matrix_dtype"]
        or (counts < 0).any()
    ):
        raise ValueError("invalid routing count matrix")
    world_size, sp_size = metadata["world_size"], metadata["sp_size"]
    num_local_experts = metadata["num_local_experts"]
    if (
        len(ranks) != ep_size
        or len(set(ranks)) != ep_size
        or ranks[0] != metadata["ep_group_leader"]
        or not all(0 <= rank < world_size for rank in ranks)
        or not 1 <= sp_size <= world_size
        or world_size % sp_size
        or world_size % ep_size
        or len(owners) != num_experts
        or set(ownership) != set(ranks)
        or any(ownership[rank] != num_local_experts for rank in ranks)
        or metadata["hidden_size"] < 1
        or metadata["bytes_per_element"] < 1
    ):
        raise ValueError("invalid parallel layout or activation size")
    return counts


def _call_key(metadata: dict) -> tuple:
    fqn, path, call = metadata["layer_fqn"], metadata["checkpoint_path"], metadata["layer_call"]
    if (
        not isinstance(fqn, str)
        or not isinstance(path, list)
        or not all(type(index) is int and index >= 0 for index in path)
        or type(call) is not int
        or call < 0
    ):
        raise ValueError("invalid checkpoint call identity")
    return fqn, tuple(path), call


def _sort_key(metadata: dict) -> tuple:
    layer_id = metadata["layer_id"]
    return (isinstance(layer_id, str), layer_id, *_call_key(metadata))


def _write_load_matrix(
    directory: Path, name: str, layers: list[dict], loads: list[np.ndarray], columns: list[str],
) -> np.ndarray:
    # Use -1 in NPY and blank in CSV for columns absent from smaller layers.
    matrix = np.full((len(layers), len(columns)), -1, dtype=np.int64)
    with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer_id", "layer_fqn", "checkpoint_path", "layer_call", "matrix_path", *columns])
        for index, (metadata, load) in enumerate(zip(layers, loads, strict=True)):
            matrix[index, :load.size] = load
            writer.writerow([
                metadata["layer_id"], metadata["layer_fqn"], json.dumps(metadata["checkpoint_path"]),
                metadata["layer_call"], metadata["matrix_path"], *load.tolist(),
                *([""] * (len(columns) - load.size)),
            ])
    np.save(directory / f"{name}.npy", matrix, allow_pickle=False)
    return matrix


def _plots(directory: Path, expert_load: np.ndarray, rank_load: np.ndarray, layers: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, load in (("expert", expert_load), ("rank", rank_load)):
        fig, ax = plt.subplots()
        heatmap = ax.imshow(np.ma.masked_less(load, 0), aspect="auto", interpolation="nearest")
        fig.colorbar(heatmap, ax=ax, label="Routed assignments")
        ax.set(xlabel=f"{name.capitalize()} column", ylabel="Layer row in CSV")
        fig.savefig(directory / f"{name}_load.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    series = defaultdict(dict)
    for row, layer in enumerate(layers):
        for simulation in layer["ep_simulations"]:
            if "rank_stats" in simulation:
                series[simulation["ep_size"]][row] = simulation["rank_stats"]["max"]
    if series:
        fig, ax = plt.subplots()
        ax.plot([layer["rank_stats"]["max"] for layer in layers], label="Measured")
        for ep_size, values in sorted(series.items()):
            ax.plot([values.get(row, np.nan) for row in range(len(layers))],
                    label=f"EP{ep_size} (fixed-routing simulation)")
        ax.set(xlabel="Layer row in CSV", ylabel="Max routed assignments per rank")
        ax.legend()
        fig.savefig(directory / "ep_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _summarize_ep_layer(summary: dict[tuple[int, str, int], dict], layer: dict) -> None:
    source_ep_size = layer["ep_size"]
    measured = {
        "ep_size": source_ep_size,
        "kind": "measured",
        "rank_stats": layer["rank_stats"],
        "max_subgroup_imbalance": layer["rank_stats"]["max_over_mean"],
        "buffer_estimates": layer["buffer_estimates"],
    }
    for result in (measured, *layer["ep_simulations"]):
        if "skipped" in result or result.get("identity"):
            continue
        ep_size, kind = result["ep_size"], result["kind"]
        row = summary.setdefault(
            (ep_size, kind, source_ep_size),
            {
                "ep_size": ep_size,
                "kind": kind,
                "source_ep_size": source_ep_size,
                "num_layer_calls": 0,
                "max_subgroup_imbalance": None,
                "max_fc2_output_bytes": 0,
            },
        )
        row["num_layer_calls"] += 1
        rank_stats = result["rank_stats"]
        if rank_stats["max"] > row.get("max_rank_tokens", -1):
            row.update(
                max_rank_tokens=rank_stats["max"],
                layer_id=layer["layer_id"],
                layer_fqn=layer["layer_fqn"],
                hot_global_rank=min(rank_stats["hot_global_ranks"]),
                source_ep_group_leader=layer["ep_group_leader"],
                microstep=layer["microstep"],
                phase=layer["phase"],
                run_id=layer.get("run_id"),
            )

        # Peak rank load, subgroup imbalance and FC2 bytes can occur in different layers.
        imbalance = result["max_subgroup_imbalance"]
        if imbalance is not None:
            row["max_subgroup_imbalance"] = max(row["max_subgroup_imbalance"] or 0.0, imbalance)
        row["max_fc2_output_bytes"] = max(
            row["max_fc2_output_bytes"], result["buffer_estimates"]["fc2_output_max_bytes"],
        )
        row["max_fc2_output_gib"] = row["max_fc2_output_bytes"] / (1 << 30)


def _write_ep_summary(path: Path, rows: Iterable[dict]) -> None:
    columns = [
        "ep_size", "max_rank_tokens", "layer_id", "hot_global_rank", "max_subgroup_imbalance",
        "max_fc2_output_bytes", "max_fc2_output_gib", "kind", "source_ep_size",
        "source_ep_group_leader", "microstep", "phase", "run_id", "layer_fqn", "num_layer_calls",
    ]
    sorted_rows = sorted(
        rows, key=lambda row: (-row["ep_size"], row["kind"] != "measured", -row["source_ep_size"]),
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted_rows)


def analyze(
    input_dir: Path, output_dir: Path, ep_sizes: list[int], plots: bool = False,
) -> dict:
    """Write reports, tolerating interrupted writes and missing microsteps."""
    input_dir = input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "num_matrices": 0,
        "skipped": [],
        "microsteps": [],
    }
    ep_summary: dict[tuple[int, str, int], dict] = {}
    for manifest in sorted(input_dir.rglob("metadata.jsonl")):
        directory = manifest.parent
        relative = directory.relative_to(input_dir)
        microsteps = defaultdict(list)
        for metadata in _read_metadata(manifest, report["skipped"]):
            microsteps[metadata["microstep"]].append(metadata)

        for microstep, records in sorted(microsteps.items()):
            entry = {"directory": str(relative), "microstep": microstep, "phases": {}, "comparisons": []}
            forward_records = {}
            recompute_keys = set()
            unmatched_recompute = 0
            records.sort(key=_sort_key)
            for phase in ("forward", "recompute"):
                layers, expert_loads, rank_loads = [], [], []
                for metadata in records:
                    if metadata["phase"] != phase:
                        continue
                    try:
                        counts = _load_matrix(directory, metadata)
                    except (OSError, EOFError, KeyError, ValueError, TypeError) as exc:
                        report["skipped"].append({
                            "manifest": str(manifest), "matrix_path": metadata.get("matrix_path"),
                            "reason": str(exc),
                        })
                        continue
                    ranks = metadata["ep_group_global_ranks"]
                    expert_load, rank_load = aggregate_load(counts, ranks, metadata["expert_placement"])
                    stats = {
                        **metadata,
                        "expert_stats": _stats(expert_load),
                        "rank_stats": _rank_stats(rank_load, ranks),
                        "buffer_estimates": _buffer_bytes(
                            rank_load, counts, metadata, metadata["num_local_experts"],
                        ),
                        "ep_simulations": [],
                    }
                    for ep_size in ep_sizes:
                        try:
                            simulation = simulate_ep(counts, metadata, ep_size)
                        except ValueError as exc:
                            simulation = {
                                "kind": "counterfactual_fixed_routing",
                                "ep_size": ep_size,
                                "skipped": str(exc),
                            }
                        stats["ep_simulations"].append(simulation)
                    _summarize_ep_layer(ep_summary, stats)

                    call_key = _call_key(metadata)
                    if phase == "forward":
                        forward_records[call_key] = metadata
                    elif call_key in forward_records:
                        forward = forward_records[call_key]
                        comparison = {
                            "forward_matrix_path": forward["matrix_path"],
                            "recompute_matrix_path": metadata["matrix_path"],
                            "layer_fqn": metadata["layer_fqn"],
                            "layer_id": metadata["layer_id"],
                            "checkpoint_path": metadata["checkpoint_path"],
                            "layer_call": metadata["layer_call"],
                        }
                        if all(forward[field] == metadata[field] for field in (
                            "matrix_shape", "ep_group_global_ranks", "expert_placement",
                        )):
                            comparison.update(compare_phases(_load_matrix(directory, forward), counts, metadata))
                        else:
                            comparison["skipped"] = "forward and recompute layouts differ"
                        entry["comparisons"].append(comparison)
                        recompute_keys.add(call_key)
                    else:
                        unmatched_recompute += 1

                    layers.append(stats)
                    expert_loads.append(expert_load)
                    rank_loads.append(rank_load)
                if not layers:
                    continue

                phase_dir = output_dir / relative / f"micro_{microstep:06d}" / phase
                phase_dir.mkdir(parents=True, exist_ok=True)
                expert_columns = [f"expert_{expert}" for expert in range(max(load.size for load in expert_loads))]
                rank_columns = [f"rank_{rank}" for rank in layers[0]["ep_group_global_ranks"]]
                expert_matrix = _write_load_matrix(phase_dir, "expert_load", layers, expert_loads, expert_columns)
                rank_matrix = _write_load_matrix(phase_dir, "rank_load", layers, rank_loads, rank_columns)
                if plots:
                    _plots(phase_dir, expert_matrix, rank_matrix, layers)
                entry["phases"][phase] = layers
                report["num_matrices"] += len(layers)
            if entry["phases"]:
                entry["forward_without_recompute"] = len(forward_records.keys() - recompute_keys)
                entry["recompute_without_forward"] = unmatched_recompute
                report["microsteps"].append(entry)

    with (output_dir / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    _write_ep_summary(output_dir / "ep_summary.csv", ep_summary.values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing MoE route dumps")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for JSON, CSV and NPY output")
    parser.add_argument("--simulate-ep", type=int, nargs="+", default=[], help="EP sizes for fixed-route simulation")
    parser.add_argument("--plots", action="store_true", help="Write PNG plots (requires Matplotlib)")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if any(ep_size < 1 for ep_size in args.simulate_ep):
        parser.error("--simulate-ep values must be positive")
    report = analyze(args.input_dir, args.output_dir, list(dict.fromkeys(args.simulate_ep)), args.plots)
    print(f"Analyzed {report['num_matrices']} matrices, skipped {len(report['skipped'])}.")
    print(args.output_dir / "report.json")
    print(args.output_dir / "ep_summary.csv")


if __name__ == "__main__":
    main()
