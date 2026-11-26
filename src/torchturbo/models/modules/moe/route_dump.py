"""Write pre-dispatch MoE routing counts on the EP group leader."""

from __future__ import annotations

import functools
import json
import os
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from torchturbo.logging import get_logger

logger = get_logger(__name__)
_CURRENT_ROUTE_CALL: ContextVar[_RouteCall | None] = ContextVar(
    "moe_route_call", default=None,
)


@dataclass
class _RouteCall:
    dump: MoERouteDump
    microstep: int
    optimizer_step: int | None
    phase: str = "forward"
    checkpoint_path: tuple[int, ...] = ()
    layer_calls: Counter[str] = field(default_factory=Counter)
    child_checkpoints: Iterator[int] = field(default_factory=count)

    @contextmanager
    def active(self) -> Iterator[None]:
        token = _CURRENT_ROUTE_CALL.set(self)
        try:
            yield
        finally:
            _CURRENT_ROUTE_CALL.reset(token)


@dataclass(frozen=True)
class RouteCheckpoint:
    """The logical checkpoint invocation shared by forward and replay."""

    parent: _RouteCall
    path: tuple[int, ...]

    def forward_context(self) -> AbstractContextManager[None]:
        return self._context(self.parent.phase)

    def recompute_context(self) -> AbstractContextManager[None]:
        return self._context("recompute")

    @contextmanager
    def _context(self, phase: str) -> Iterator[None]:
        call = _RouteCall(
            self.parent.dump, self.parent.microstep, self.parent.optimizer_step,
            phase, self.path,
        )
        with call.active():
            yield


def capture_route_checkpoint() -> RouteCheckpoint | None:
    """Capture the current microstep in a checkpoint's closure."""
    parent = _CURRENT_ROUTE_CALL.get()
    if parent is None:
        return None
    return RouteCheckpoint(parent, (*parent.checkpoint_path, next(parent.child_checkpoints)))


class MoERouteDump:
    """Own the route dump for one prepared model on an EP group leader."""

    def __init__(
        self, directory: Path, phase: str, max_microsteps: int,
        ep_ranks: list[int], world_size: int, sp_size: int,
    ) -> None:
        self.run_id = uuid4().hex
        self.directory = directory / f"ep_group_{ep_ranks[0]}" / f"run_{self.run_id}"
        self.ep_ranks = ep_ranks
        self.world_size = world_size
        self.sp_size = sp_size
        self.phase = phase
        self.max_microsteps = max_microsteps
        self.microsteps = count()
        self.optimizer_step: int | None = None
        self.failed = False

    def bind_optimizer(self, optimizer: Optimizer) -> None:
        """Track completed optimizer.step calls for this process."""
        self.optimizer_step = 0
        steps = count(1)

        def after_step(_optimizer, _args, _kwargs):
            self.optimizer_step = next(steps)

        optimizer.register_step_post_hook(after_step)

    def wrap_forward(self, model: nn.Module) -> None:
        """Give each training root forward a zero-based microstep."""
        original_forward = model.forward

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            if not model.training or self.failed:
                return original_forward(*args, **kwargs)
            microstep = next(self.microsteps)
            if microstep >= self.max_microsteps:
                return original_forward(*args, **kwargs)
            call = _RouteCall(self, microstep, self.optimizer_step)
            with call.active():
                return original_forward(*args, **kwargs)

        model.forward = forward

    def _write(self, metadata: dict, counts: np.ndarray, layer_stem: str) -> None:
        if int(counts.max()) <= np.iinfo(np.int32).max:
            counts = counts.astype(np.int32, copy=False)

        phase_dir = self.directory / f"micro_{metadata['microstep']:06d}" / metadata["phase"]
        try:
            phase_dir.mkdir(parents=True, exist_ok=True)
            path = phase_dir / f"{layer_stem}.npy"
            # MTP and decoder layers can share a layer ID, and a layer can run twice.
            for suffix in count(1):
                if not path.exists():
                    break
                path = phase_dir / f"{layer_stem}_{suffix}.npy"
            metadata.update(
                matrix_shape=list(counts.shape),
                matrix_dtype=counts.dtype.name,
                matrix_path=path.relative_to(self.directory).as_posix(),
            )
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, counts, allow_pickle=False)
            # A killed process can leave a manifest row without its final file.
            # Publishing in this order gives every final file a manifest row.
            with (self.directory / "metadata.jsonl").open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(metadata, separators=(",", ":")) + "\n")
                manifest.flush()
            temporary.replace(path)
        except OSError as exc:
            self.failed = True
            logger.warning("Disabling MoE route dumps to %s: %s", self.directory, exc)


@dataclass(frozen=True)
class LayerRouteDump:
    """Bind layer identity and actual contiguous expert placement to a dispatcher."""

    dump: MoERouteDump
    layer_id: int | str
    layer_fqn: str
    num_local_experts: int

    def record(self, global_counts: torch.Tensor, hidden_states: torch.Tensor) -> None:
        """Publish the gathered counts before any token permutation or exchange."""
        call = _CURRENT_ROUTE_CALL.get()
        if call is None or call.dump is not self.dump or self.dump.failed:
            return
        if self.dump.phase not in ("both", call.phase):
            return
        layer_call = call.layer_calls[self.layer_fqn]
        call.layer_calls[self.layer_fqn] += 1
        ep_ranks = self.dump.ep_ranks
        metadata = {
            "schema_version": 1,
            "run_id": self.dump.run_id,
            "optimizer_step": call.optimizer_step,
            "microstep": call.microstep,
            "layer_id": self.layer_id,
            "layer_fqn": self.layer_fqn,
            "layer_call": layer_call,
            "checkpoint_path": list(call.checkpoint_path),
            "phase": call.phase,
            "world_size": self.dump.world_size,
            "sp_size": self.dump.sp_size,
            "ep_size": len(ep_ranks),
            "ep_group_leader": ep_ranks[0],
            "ep_group_global_ranks": ep_ranks,
            "num_experts": global_counts.shape[1],
            "num_local_experts": self.num_local_experts,
            "expert_placement": [rank for rank in ep_ranks for _ in range(self.num_local_experts)],
            "hidden_size": hidden_states.shape[-1],
            "compute_dtype": str(hidden_states.dtype).removeprefix("torch."),
            "compute_dtype_basis": "dispatcher_input",
            "bytes_per_element": hidden_states.element_size(),
        }
        layer_stem = f"layer_{self.layer_id:03d}" if isinstance(self.layer_id, int) else "layer"
        # Finish the D2H copy and publish the file before dispatch can allocate.
        self.dump._write(metadata, global_counts.detach().cpu().numpy(), layer_stem)


def install_moe_route_dump(model: nn.Module, sp_size: int) -> MoERouteDump | None:
    """Attach diagnostics to the prepared TorchTurbo MoE layers when requested."""
    if os.environ.get("TT_MOE_DUMP_ENABLED") != "1":
        return None

    from .moe_layer import BaseMoELayer
    from .token_dispatcher import MoEAlltoAllTokenDispatcher

    phase = os.environ.get("TT_MOE_DUMP_PHASE", "forward")
    if phase not in ("forward", "recompute", "both"):
        raise ValueError("TT_MOE_DUMP_PHASE must be forward, recompute, or both")
    max_microsteps = int(os.environ.get("TT_MOE_DUMP_MAX_MICROSTEPS", "4"))
    if max_microsteps < 1:
        raise ValueError("TT_MOE_DUMP_MAX_MICROSTEPS must be positive")

    layers = [
        (fqn, module)
        for fqn, module in model.named_modules()
        if isinstance(module, BaseMoELayer)
        and isinstance(module.token_dispatcher, MoEAlltoAllTokenDispatcher)
    ]
    if not layers:
        logger.warning_rank0("MoE route dump enabled but no TorchTurbo all-to-all dispatchers were found.")
        return None

    # FSDP2 uses one EP mesh dimension for all TorchTurbo MoE layers.
    ep_group = layers[0][1].token_dispatcher.ep_group
    if dist.get_rank(ep_group) != 0:
        return None
    dump = MoERouteDump(
        Path(os.environ.get("TT_MOE_DUMP_DIR", "moe_diagnostics")),
        phase, max_microsteps, dist.get_process_group_ranks(ep_group),
        dist.get_world_size(), sp_size,
    )
    for fqn, module in layers:
        dispatcher = module.token_dispatcher
        layer_id = module.layer_number
        if layer_id is None:
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", fqn)
            layer_id = int(match[1]) if match else fqn or "<root>"
        dispatcher.route_dump = LayerRouteDump(
            dump, layer_id, fqn, dispatcher.num_local_experts,
        )
    dump.wrap_forward(model)
    return dump
