"""CPU checks for actual and fixed-routing counterfactual loads."""

import csv
import json

import numpy as np
import pytest

from tools.analyze_moe_load import aggregate_load, analyze, compare_phases, simulate_ep


def _metadata():
    ranks = list(range(128, 256))
    return {
        "ep_group_global_ranks": ranks,
        "world_size": 256,
        "sp_size": 128,
        "num_local_experts": 2,
        "expert_placement": [rank for rank in ranks for _ in range(2)],
        "hidden_size": 8192,
        "bytes_per_element": 2,
    }


def test_expert_and_rank_load_use_actual_placement():
    counts = np.array([[1, 2, 5, 4], [7, 11, 13, 3]], dtype=np.int32)
    expert, rank = aggregate_load(counts, [7, 3], [3, 7, 3, 7])
    np.testing.assert_array_equal(expert, [8, 13, 18, 7])
    np.testing.assert_array_equal(rank, [20, 26])


@pytest.mark.parametrize("ep_size", [64, 32])
@pytest.mark.parametrize("case,expected", [
    ("balanced", 0),
    ("all_zero", 0),
    ("distributed_hotspot", None),
    ("one_source", 0),
    ("neighbor_experts", 7),
])
def test_ep128_split_can_improve_preserve_or_worsen_rank_load(ep_size, case, expected):
    metadata = _metadata()
    counts = np.zeros((128, 256), dtype=np.int32)
    if case == "balanced":
        counts.fill(1)
    elif case == "distributed_hotspot":
        counts[:, 0] = 10
        expected = -10 * (128 - ep_size)
    elif case == "one_source":
        counts[0, 0] = 10
    elif case == "neighbor_experts":
        counts[0, 0] = 7
        counts[1, 2] = 10

    result = simulate_ep(counts, metadata, ep_size)
    assert result["kind"] == "counterfactual_fixed_routing"
    assert result["max_rank_load_delta"] == expected
    assert result["num_local_experts"] == 256 // ep_size
    subgroups = result["subgroups"]
    assert len(subgroups) == 128 // ep_size
    for offset, subgroup in enumerate(subgroups):
        start = offset * ep_size
        assert subgroup["global_ranks"] == metadata["ep_group_global_ranks"][start:start + ep_size]
        assert sum(subgroup["rank_load"]) == counts[start:start + ep_size].sum()
    stats = result["rank_stats"]
    if case == "balanced":
        assert stats["min"] == stats["max"] == stats["mean"] == 256
        assert stats["max_over_mean"] == 1
        assert result["max_subgroup_imbalance"] == 1
    elif case == "all_zero":
        assert stats["max_over_mean"] is None
        assert result["max_subgroup_imbalance"] is None
    elif case == "distributed_hotspot":
        assert stats["max_over_mean"] == ep_size
        assert stats["hot_global_ranks"] == [group["global_ranks"][0] for group in subgroups]
        assert result["buffer_estimates"] == {
            "fc2_output_max_bytes": 10 * ep_size * 8192 * 2,
            "combine_permutation_max_bytes": 10 * ep_size * 8192 * 2,
            "combine_returned_packed_max_bytes": 10 * 8192 * 2,
        }


def test_subgroup_imbalance_uses_its_own_mean():
    counts = np.array([[20, 0, 0, 0], [0, 0, 0, 0], [100, 0, 0, 0], [0, 0, 100, 0]], dtype=np.int32)
    metadata = {
        **_metadata(), "ep_group_global_ranks": [4, 5, 6, 7],
        "expert_placement": [4, 5, 6, 7], "world_size": 8, "sp_size": 1,
    }
    result = simulate_ep(counts, metadata, 2)

    first, second = result["subgroups"]
    assert first["rank_load"] == [20, 0]
    assert first["rank_stats"]["mean"] == 10
    assert first["rank_stats"]["max_over_mean"] == 2
    assert second["rank_load"] == [100, 100]
    assert second["rank_stats"]["mean"] == 100
    assert second["rank_stats"]["max_over_mean"] == 1
    assert result["rank_stats"]["max_over_mean"] == pytest.approx(100 / 55)
    assert result["max_subgroup_imbalance"] == 2

    counts[:2] = 0
    result = simulate_ep(counts, metadata, 2)
    assert result["subgroups"][0]["rank_stats"]["max_over_mean"] is None
    assert result["max_subgroup_imbalance"] == 1


def test_ep_summary_aggregates_groups_microsteps_and_layers(tmp_path):
    input_dir, output_dir = tmp_path / "dumps", tmp_path / "analysis"

    def write_layer(leader, microstep, phase, layer, hidden_size, source_rows):
        run_id = f"run_{leader}"
        directory = input_dir / f"ep_group_{leader}" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        counts = np.zeros((4, 4), dtype=np.int32)
        for source, expert, tokens in source_rows:
            counts[source, expert] = tokens
        matrix_path = f"{microstep}_{phase}_{layer}.npy"
        np.save(directory / matrix_path, counts, allow_pickle=False)
        ranks = list(range(leader, leader + 4))
        metadata = {
            "schema_version": 1, "run_id": run_id, "microstep": microstep, "phase": phase,
            "layer_id": layer, "layer_fqn": f"layers.{layer}.mlp", "checkpoint_path": [], "layer_call": 0,
            "matrix_path": matrix_path, "matrix_shape": [4, 4], "matrix_dtype": "int32",
            "world_size": 8, "sp_size": 1, "ep_size": 4,
            "ep_group_global_ranks": ranks, "ep_group_leader": leader, "expert_placement": ranks,
            "num_experts": 4, "num_local_experts": 1, "hidden_size": hidden_size, "bytes_per_element": 2,
        }
        with (directory / "metadata.jsonl").open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(metadata) + "\n")

    write_layer(0, 0, "forward", 0, 8192, [(0, 0, 40), (0, 2, 1)])
    write_layer(0, 0, "forward", 42, 16, [(0, expert, 30) for expert in range(4)])
    write_layer(4, 1, "recompute", 35, 16, [(0, 0, 20), (2, 0, 100), (3, 2, 100)])
    report = analyze(input_dir, output_dir, [2, 3, 4])

    assert report["num_matrices"] == 3
    with (output_dir / "ep_summary.csv").open(newline="", encoding="utf-8") as stream:
        measured, simulated = csv.DictReader(stream)
    assert (measured["ep_size"], measured["kind"]) == ("4", "measured")
    assert measured["max_rank_tokens"] == "120"
    assert measured["hot_global_rank"] == "4"
    assert float(measured["max_subgroup_imbalance"]) == pytest.approx(40 / (41 / 4))
    assert (simulated["ep_size"], simulated["kind"]) == ("2", "counterfactual_fixed_routing")
    assert simulated["source_ep_size"] == "4"
    assert simulated["max_rank_tokens"] == "100"
    assert simulated["layer_id"] == "35"
    assert simulated["hot_global_rank"] == "6"
    assert simulated["source_ep_group_leader"] == "4"
    assert simulated["microstep"] == "1"
    assert simulated["phase"] == "recompute"
    assert simulated["num_layer_calls"] == "3"
    assert float(simulated["max_subgroup_imbalance"]) == 2
    assert int(simulated["max_fc2_output_bytes"]) == 40 * 8192 * 2
    assert float(simulated["max_fc2_output_gib"]) == pytest.approx(40 * 8192 * 2 / (1 << 30))


def test_ep_identity_and_invalid_mesh():
    counts = np.ones((128, 256), dtype=np.int32)
    metadata = _metadata()
    assert simulate_ep(counts, metadata, 128)["identity"]
    for ep_size in (0, 3, 256):
        with pytest.raises(ValueError, match="EP must divide"):
            simulate_ep(counts, metadata, ep_size)
    metadata["ep_group_global_ranks"].reverse()
    with pytest.raises(ValueError, match="contiguous world mesh"):
        simulate_ep(counts, metadata, 32)


def test_phase_comparison_detects_source_changes_even_if_loads_are_equal():
    forward = np.zeros((128, 256), dtype=np.int32)
    forward[0, 0] = 10
    recompute = forward.copy()
    recompute[0, 0] = 0
    recompute[1, 0] = 10
    result = compare_phases(forward, recompute, _metadata())
    assert result["num_changed_elements"] == 2
    assert result["max_abs_count_delta"] == 10
    assert result["max_abs_rank_load_delta"] == 0

    recompute[1, 0] = 3
    recompute[1, 2] = 7
    result = compare_phases(forward, recompute, _metadata())
    assert result["rank_load_delta"][:2] == [-7, 7]
    assert result["max_abs_rank_load_delta"] == 7
