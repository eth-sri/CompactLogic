from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any

from .gates import GATE_NAMES


@dataclass(frozen=True)
class GateInput:
    """One gate operand source: either another signal or a compile-time constant."""

    signal_id: int | None = None
    const_value: int | None = None

    def __post_init__(self) -> None:
        has_signal = self.signal_id is not None
        has_const = self.const_value is not None
        if has_signal == has_const:
            raise ValueError('Exactly one of signal_id or const_value must be set.')
        if self.const_value is not None and self.const_value not in (0, 1):
            raise ValueError(f'Invalid const_value: {self.const_value}')

    @property
    def is_const(self) -> bool:
        return self.const_value is not None


@dataclass(frozen=True)
class GateNode:
    """One discretized 2-input Boolean gate in the compiled circuit."""

    node_id: int
    layer_index: int
    local_index: int
    gate_id: int
    input_a: GateInput
    input_b: GateInput

    @property
    def gate_name(self) -> str:
        return GATE_NAMES[self.gate_id]


@dataclass(frozen=True)
class LogicLayerIR:
    """A regular dense Boolean layer in compiled form."""

    layer_index: int
    in_dim: int
    out_dim: int
    input_start: int
    output_start: int
    nodes: tuple[GateNode, ...]

    @property
    def output_stop(self) -> int:
        return self.output_start + self.out_dim


@dataclass(frozen=True)
class GroupSumIR:
    """Compiled view of the final GroupSum decoder."""

    num_classes: int
    tau: float
    input_start: int
    input_dim: int

    @property
    def padded_input_dim(self) -> int:
        return ceil(self.input_dim / self.num_classes) * self.num_classes

    @property
    def group_size(self) -> int:
        return self.padded_input_dim // self.num_classes

    def groups(self) -> tuple[tuple[int | None, ...], ...]:
        groups: list[tuple[int | None, ...]] = []
        for class_idx in range(self.num_classes):
            base = self.input_start + class_idx * self.group_size
            entries = []
            for offset in range(self.group_size):
                local_index = class_idx * self.group_size + offset
                if local_index < self.input_dim:
                    entries.append(base + offset)
                else:
                    entries.append(None)
            groups.append(tuple(entries))
        return tuple(groups)


@dataclass(frozen=True)
class RegularCircuitIR:
    """Hardware-oriented IR for an extracted compactlogic Boolean circuit."""

    input_dim: int
    layers: tuple[LogicLayerIR, ...]
    output: GroupSumIR
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return sum(layer.out_dim for layer in self.layers)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def total_signal_count(self) -> int:
        return self.input_dim + self.node_count

    def gate_histogram(self) -> dict[str, int]:
        counts = {name: 0 for name in GATE_NAMES}
        for layer in self.layers:
            for node in layer.nodes:
                counts[node.gate_name] += 1
        return {name: count for name, count in counts.items() if count > 0}

    def summary(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "layer_count": self.layer_count,
            "node_count": self.node_count,
            "total_signal_count": self.total_signal_count,
            "output_classes": self.output.num_classes,
            "group_size": self.output.group_size,
            "tau": self.output.tau,
            "layers": [
                {
                    "layer_index": layer.layer_index,
                    "in_dim": layer.in_dim,
                    "out_dim": layer.out_dim,
                    "input_start": layer.input_start,
                    "output_start": layer.output_start,
                }
                for layer in self.layers
            ],
            "gate_histogram": self.gate_histogram(),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "layers": [
                {
                    "layer_index": layer.layer_index,
                    "in_dim": layer.in_dim,
                    "out_dim": layer.out_dim,
                    "input_start": layer.input_start,
                    "output_start": layer.output_start,
                    "nodes": [asdict(node) for node in layer.nodes],
                }
                for layer in self.layers
            ],
            "output": {
                "num_classes": self.output.num_classes,
                "tau": self.output.tau,
                "input_start": self.output.input_start,
                "input_dim": self.output.input_dim,
                "group_size": self.output.group_size,
                "groups": self.output.groups(),
            },
            "metadata": self.metadata,
        }


CompiledCircuitIR = RegularCircuitIR
