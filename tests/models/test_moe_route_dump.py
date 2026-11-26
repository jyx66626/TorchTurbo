"""CPU coverage for pre-dispatch persistence and interrupted runs."""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.prefetch import FSDPPrefetchManager
from torchturbo.engine.config import EngineConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe import token_dispatcher as dispatch_module
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import BaseMoELayer, MoELayer, MoESubmodules
from torchturbo.models.modules.moe.route_dump import (
    LayerRouteDump,
    MoERouteDump,
    install_moe_route_dump,
)
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
from tools.analyze_moe_load import analyze


def _dispatcher(monkeypatch, ep_size=2):
    monkeypatch.setattr(dispatch_module, "get_device_runtime", CpuRuntime)
    return dispatch_module.MoEAlltoAllTokenDispatcher(
        2, [0, 1], MoEConfig(
            num_moe_experts=ep_size * 2, expert_model_parallel_size=ep_size,
            moe_router_topk=1,
        ), object(),
    )


@pytest.mark.parametrize("maximum", [np.iinfo(np.int32).max, np.iinfo(np.int32).max + 1])
def test_store_raw_matrix_with_safe_count_dtype(tmp_path, maximum):
    dump = MoERouteDump(tmp_path, "forward", 1, [2, 3], 4, 4)
    layer = LayerRouteDump(dump, 18, "layers.18.mlp", 1)
    counts = torch.tensor([[maximum, 3], [0, 9]])

    class Model(nn.Module):
        def forward(self, hidden_states):
            layer.record(counts, hidden_states)

    model = Model()
    dump.wrap_forward(model)
    model(torch.zeros((2, 16), dtype=torch.bfloat16))
    model(torch.zeros((2, 16), dtype=torch.bfloat16))
    (row,) = (dump.directory / "metadata.jsonl").read_text().splitlines()
    metadata = json.loads(row)
    matrix = np.load(dump.directory / metadata["matrix_path"], allow_pickle=False)
    np.testing.assert_array_equal(matrix, counts.numpy())
    assert matrix.dtype.name == ("int32" if maximum == np.iinfo(np.int32).max else "int64")
    assert metadata["matrix_dtype"] == matrix.dtype.name
    assert metadata["layer_id"] == 18
    assert metadata["ep_group_global_ranks"] == [2, 3]
    assert metadata["expert_placement"] == [2, 3]
    assert metadata["compute_dtype"] == "bfloat16"
    assert metadata["hidden_size"] == 16
    assert metadata["bytes_per_element"] == 2


def test_publishes_current_layer_before_permutation_oom(tmp_path, monkeypatch):
    dispatcher = _dispatcher(monkeypatch)
    counts = torch.tensor([[2, 1, 0, 0], [0, 4, 1, 0]])
    gathers = []

    def gather(local_counts, group):
        gathers.append(group)
        torch.testing.assert_close(local_counts, counts[0])
        return counts.flatten()

    monkeypatch.setattr(dispatch_module, "gather_from_sequence_parallel_region", gather)
    dump = MoERouteDump(tmp_path, "forward", 2, [0, 1], 2, 1)
    layers = [LayerRouteDump(dump, i, f"layers.{i}.mlp", 2) for i in range(2)]
    routing_map = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)

    class Model(nn.Module):
        def forward(self, hidden_states):
            dispatcher.route_dump = layers[0]
            dispatcher.preprocess(routing_map, hidden_states)
            dispatcher.route_dump = layers[1]
            dispatcher.dispatch_preprocess(hidden_states, routing_map, routing_map.float())

    class OOMOps:
        def moe_permutation(self, *args, **kwargs):
            raise torch.OutOfMemoryError("simulated pre-expert OOM")

    monkeypatch.setattr(dispatch_module, "ops", OOMOps())
    model = Model()
    dump.wrap_forward(model)
    with pytest.raises(torch.OutOfMemoryError):
        model(torch.zeros((3, 16)))
    assert gathers == [dispatcher.ep_group, dispatcher.ep_group]

    # Model an interruption after the manifest flush and before the atomic rename,
    # a truncated final NPY, and an incomplete final manifest line.
    manifest = dump.directory / "metadata.jsonl"
    rows = [json.loads(row) for row in manifest.read_text().splitlines()]
    partial_matrix = "micro_000000/forward/layer_002.npy"
    broken_matrix = "micro_000000/forward/layer_003.npy"
    (dump.directory / (partial_matrix + ".tmp")).write_bytes(b"partial")
    (dump.directory / broken_matrix).write_bytes(b"\x93NUMPY")
    with manifest.open("a", encoding="utf-8") as stream:
        for layer_id, path in enumerate((partial_matrix, broken_matrix), start=2):
            stream.write(json.dumps({**rows[-1], "layer_id": layer_id, "matrix_path": path}) + "\n")
        stream.write('{"schema_version":')

    report = analyze(tmp_path, tmp_path / "analysis", [1, 2])
    assert report["num_matrices"] == 2
    assert len(report["skipped"]) == 3
    for row in rows:
        matrix = np.load(dump.directory / row["matrix_path"], allow_pickle=False)
        np.testing.assert_array_equal(matrix, counts.numpy())
    assert report["microsteps"][0]["phases"]["forward"][1]["rank_stats"]["max"] == 7


@pytest.mark.parametrize("enabled,ep_rank", [(False, 0), (True, 1)])
def test_only_enabled_ep_leader_instruments_dispatch(tmp_path, monkeypatch, enabled, ep_rank):
    dispatcher = _dispatcher(monkeypatch)
    monkeypatch.setenv("TT_MOE_DUMP_ENABLED", str(int(enabled)))
    monkeypatch.setenv("TT_MOE_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: ep_rank)
    gathered = torch.tensor([1, 0, 0, 1, 2, 4, 8, 16])
    gather_calls = []

    def gather(*args, **kwargs):
        gather_calls.append(1)
        return gathered

    def forbid_d2h(*args, **kwargs):
        pytest.fail("Inactive diagnostics must not copy the gathered matrix.")

    monkeypatch.setattr(dispatch_module, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(torch.Tensor, "cpu", forbid_d2h)

    class Layer(BaseMoELayer):
        def __init__(self):
            nn.Module.__init__(self)
            self.token_dispatcher = dispatcher

        def forward(self, hidden_states):
            return dispatcher.preprocess(torch.ones((1, 4), dtype=torch.bool), hidden_states)

    model = Layer()
    forward = model.forward
    assert install_moe_route_dump(model, sp_size=1) is None
    assert model.forward == forward
    torch.testing.assert_close(model(torch.zeros((1, 8))), torch.tensor([3, 4]))
    assert len(gather_calls) == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("dump_phase", [None, "recompute", "both"], ids=["default", "recompute", "both"])
def test_route_dump_pairs_checkpoint_replays(
    tmp_path, monkeypatch, use_reentrant, dump_phase,
):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0])
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
    monkeypatch.setenv("TT_MOE_DUMP_ENABLED", "1")
    monkeypatch.setenv("TT_MOE_DUMP_DIR", str(tmp_path / "dumps"))
    monkeypatch.setenv("TT_MOE_DUMP_MAX_MICROSTEPS", "3")
    if dump_phase is None:
        monkeypatch.delenv("TT_MOE_DUMP_PHASE", raising=False)
    else:
        monkeypatch.setenv("TT_MOE_DUMP_PHASE", dump_phase)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MoELayer(
                MoEConfig(
                    hidden_size=8, moe_ffn_hidden_size=4, num_moe_experts=2, moe_router_topk=1,
                ),
                object(), submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
            )])

        def forward(self, x):
            return self.layers[0](x)[0] + self.layers[0](-x)[0]

    model = Model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
        model.layers[0].router.weight[0].fill_(1)
        model.layers[0].router.weight[1].fill_(-1)
    config = EngineConfig.from_dict({
        "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
    })
    backend = SimpleNamespace(
        config=config, prefetch_manager=FSDPPrefetchManager(config.backend_config),
    )
    FSDP2Backend._apply_activation_checkpointing(
        backend,
        model,
        SimpleNamespace(modules={"layers.0": FSDPPlanSpec(activation_checkpoint=True)}),
    )
    dump = install_moe_route_dump(model, sp_size=1)
    assert dump is not None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dump.bind_optimizer(optimizer)

    def forward(length):
        return model(torch.ones(length, 8, requires_grad=True)).square().sum()

    model.eval()
    forward(1)
    model.train()
    forward(3).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (forward(5) + forward(7)).backward()
    forward(9).backward()

    rows = [
        json.loads(line)
        for line in (dump.directory / "metadata.jsonl").read_text().splitlines()
    ]
    phases = {"forward", "recompute"} if dump_phase == "both" else {dump_phase or "forward"}
    assert {row["phase"] for row in rows} == phases
    assert len(rows) == 3 * 2 * len(phases)
    assert len({row["matrix_path"] for row in rows}) == len(rows)
    assert {row["layer_id"] for row in rows} == {0}
    assert {tuple(row["checkpoint_path"]) for row in rows} == {(0,), (1,)}
    report = analyze(tmp_path / "dumps", tmp_path / "analysis", [1])
    assert report["num_matrices"] == len(rows)
    assert not report["skipped"]
    for row in rows:
        counts = np.load(dump.directory / row["matrix_path"], allow_pickle=False)
        assert counts.shape == (1, 2)
        length = (3, 5, 7)[row["microstep"]]
        expected = [[length, 0]] if row["checkpoint_path"] == [0] else [[0, length]]
        np.testing.assert_array_equal(counts, expected)
        assert row["optimizer_step"] == int(row["microstep"] > 0)
    for microstep in report["microsteps"]:
        comparisons = microstep["comparisons"]
        assert len(comparisons) == (2 if dump_phase == "both" else 0)
        assert all(comparison["num_changed_elements"] == 0 for comparison in comparisons)
