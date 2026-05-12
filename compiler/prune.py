from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gates import CONST_GATE_IDS, GATE_NAMES, truth_eval_scalar
from .ir import GateInput, GateNode, RegularCircuitIR


@dataclass(frozen=True)
class BoolSource:
    """A boolean source used by pruned outputs.

    Exactly one of `signal_id` or `const_value` must be set.
    """

    signal_id: int | None = None
    const_value: int | None = None
    inverted: bool = False

    def __post_init__(self) -> None:
        has_signal = self.signal_id is not None
        has_const = self.const_value is not None
        if has_signal == has_const:
            raise ValueError('Exactly one of signal_id or const_value must be set.')
        if self.const_value is not None and self.const_value not in (0, 1):
            raise ValueError(f'Invalid const_value: {self.const_value}')
        if self.const_value is not None and self.inverted:
            raise ValueError('Constants must be stored in non-inverted form.')


@dataclass(frozen=True)
class NaivePrunedCircuit:
    """Stage-1 pruned circuit following the paper’s simple pruning strategy."""

    input_dim: int
    nodes: tuple[GateNode, ...]
    class_groups: tuple[tuple[BoolSource, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_groups)

    @property
    def bop_count(self) -> int:
        return len(self.nodes)

    def gate_histogram(self) -> dict[str, int]:
        counts = {name: 0 for name in GATE_NAMES}
        for node in self.nodes:
            counts[node.gate_name] += 1
        return {name: count for name, count in counts.items() if count > 0}

    def summary(self) -> dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'bop_count': self.bop_count,
            'kept_node_ids': [node.node_id for node in self.nodes[:32]],
            'gate_histogram': self.gate_histogram(),
            'metadata': self.metadata,
        }


@dataclass(frozen=True)
class _ConstExpr:
    value: int


@dataclass(frozen=True)
class _RefExpr:
    signal_id: int
    inverted: bool = False


@dataclass(frozen=True)
class _GateExpr:
    signal_id: int
    gate_id: int
    input_a: GateInput
    input_b: GateInput
    layer_index: int
    local_index: int


_Expr = _ConstExpr | _RefExpr | _GateExpr

def _truth_eval(gate_id: int, a: int, b: int) -> int:
    return truth_eval_scalar(gate_id, a, b)


def _gate_id_from_table(values: tuple[int, int, int, int]) -> int:
    for gate_id in range(16):
        table = (
            _truth_eval(gate_id, 0, 0),
            _truth_eval(gate_id, 0, 1),
            _truth_eval(gate_id, 1, 0),
            _truth_eval(gate_id, 1, 1),
        )
        if table == values:
            return gate_id
    raise ValueError(f'No gate id matches truth table {values}')


def _invert_a(gate_id: int) -> int:
    return _gate_id_from_table(
        (
            _truth_eval(gate_id, 1, 0),
            _truth_eval(gate_id, 1, 1),
            _truth_eval(gate_id, 0, 0),
            _truth_eval(gate_id, 0, 1),
        )
    )


def _invert_b(gate_id: int) -> int:
    return _gate_id_from_table(
        (
            _truth_eval(gate_id, 0, 1),
            _truth_eval(gate_id, 0, 0),
            _truth_eval(gate_id, 1, 1),
            _truth_eval(gate_id, 1, 0),
        )
    )


def _collapse_const_a(gate_id: int, a_const: int, b_signal: int) -> _Expr:
    f0 = _truth_eval(gate_id, a_const, 0)
    f1 = _truth_eval(gate_id, a_const, 1)
    if f0 == 0 and f1 == 0:
        return _ConstExpr(0)
    if f0 == 1 and f1 == 1:
        return _ConstExpr(1)
    if f0 == 0 and f1 == 1:
        return _RefExpr(b_signal, inverted=False)
    return _RefExpr(b_signal, inverted=True)


def _collapse_const_b(gate_id: int, a_signal: int, b_const: int) -> _Expr:
    f0 = _truth_eval(gate_id, 0, b_const)
    f1 = _truth_eval(gate_id, 1, b_const)
    if f0 == 0 and f1 == 0:
        return _ConstExpr(0)
    if f0 == 1 and f1 == 1:
        return _ConstExpr(1)
    if f0 == 0 and f1 == 1:
        return _RefExpr(a_signal, inverted=False)
    return _RefExpr(a_signal, inverted=True)


def _collapse_same_signal(gate_id: int, signal_id: int) -> _Expr:
    f0 = _truth_eval(gate_id, 0, 0)
    f1 = _truth_eval(gate_id, 1, 1)
    if f0 == 0 and f1 == 0:
        return _ConstExpr(0)
    if f0 == 1 and f1 == 1:
        return _ConstExpr(1)
    if f0 == 0 and f1 == 1:
        return _RefExpr(signal_id, inverted=False)
    return _RefExpr(signal_id, inverted=True)


class _NaivePruner:
    def __init__(self, circuit: RegularCircuitIR):
        self.circuit = circuit
        self.node_map = {
            node.node_id: node
            for layer in circuit.layers
            for node in layer.nodes
        }
        self.memo: dict[int, _Expr] = {}
        self.live_gates: dict[int, GateNode] = {}

    def simplify_input(self, source: GateInput) -> _Expr:
        if source.const_value is not None:
            return _ConstExpr(source.const_value)
        assert source.signal_id is not None
        return self.simplify_signal(source.signal_id)

    def simplify_signal(self, signal_id: int) -> _Expr:
        if signal_id < self.circuit.input_dim:
            return _RefExpr(signal_id)
        if signal_id in self.memo:
            return self.memo[signal_id]

        node = self.node_map[signal_id]
        expr_a = self.simplify_input(node.input_a)
        expr_b = self.simplify_input(node.input_b)
        simplified = self._simplify_gate(node, expr_a, expr_b)
        self.memo[signal_id] = simplified
        return simplified

    def _simplify_gate(self, node: GateNode, expr_a: _Expr, expr_b: _Expr) -> _Expr:
        gate_id = node.gate_id

        if isinstance(expr_a, _GateExpr):
            expr_a = _RefExpr(expr_a.signal_id, inverted=False)
        if isinstance(expr_b, _GateExpr):
            expr_b = _RefExpr(expr_b.signal_id, inverted=False)

        if isinstance(expr_a, _RefExpr) and expr_a.inverted:
            gate_id = _invert_a(gate_id)
            expr_a = _RefExpr(expr_a.signal_id, inverted=False)
        if isinstance(expr_b, _RefExpr) and expr_b.inverted:
            gate_id = _invert_b(gate_id)
            expr_b = _RefExpr(expr_b.signal_id, inverted=False)

        if isinstance(expr_a, _ConstExpr) and isinstance(expr_b, _ConstExpr):
            return _ConstExpr(_truth_eval(gate_id, expr_a.value, expr_b.value))
        if isinstance(expr_a, _ConstExpr) and isinstance(expr_b, _RefExpr):
            return _collapse_const_a(gate_id, expr_a.value, expr_b.signal_id)
        if isinstance(expr_a, _RefExpr) and isinstance(expr_b, _ConstExpr):
            return _collapse_const_b(gate_id, expr_a.signal_id, expr_b.value)
        if isinstance(expr_a, _RefExpr) and isinstance(expr_b, _RefExpr):
            if expr_a.signal_id == expr_b.signal_id:
                return _collapse_same_signal(gate_id, expr_a.signal_id)
            if gate_id in CONST_GATE_IDS:
                return _ConstExpr(0 if gate_id == 0 else 1)
            if gate_id == 3:
                return _RefExpr(expr_a.signal_id)
            if gate_id == 5:
                return _RefExpr(expr_b.signal_id)
            if gate_id == 10:
                return _RefExpr(expr_b.signal_id, inverted=True)
            if gate_id == 12:
                return _RefExpr(expr_a.signal_id, inverted=True)

            gate = GateNode(
                node_id=node.node_id,
                layer_index=node.layer_index,
                local_index=node.local_index,
                gate_id=gate_id,
                input_a=GateInput(signal_id=expr_a.signal_id),
                input_b=GateInput(signal_id=expr_b.signal_id),
            )
            self.live_gates[node.node_id] = gate
            return _GateExpr(
                signal_id=node.node_id,
                gate_id=gate.gate_id,
                input_a=gate.input_a,
                input_b=gate.input_b,
                layer_index=gate.layer_index,
                local_index=gate.local_index,
            )

        raise TypeError(f'Unexpected expressions: {expr_a!r}, {expr_b!r}')

    def prune(self) -> NaivePrunedCircuit:
        class_groups: list[tuple[BoolSource, ...]] = []
        for group in self.circuit.output.groups():
            pruned_group: list[BoolSource] = []
            for signal_id in group:
                if signal_id is None:
                    pruned_group.append(BoolSource(const_value=0))
                    continue
                expr = self.simplify_signal(signal_id)
                if isinstance(expr, _ConstExpr):
                    pruned_group.append(BoolSource(const_value=expr.value))
                elif isinstance(expr, _RefExpr):
                    pruned_group.append(BoolSource(signal_id=expr.signal_id, inverted=expr.inverted))
                elif isinstance(expr, _GateExpr):
                    pruned_group.append(BoolSource(signal_id=expr.signal_id))
                else:
                    raise TypeError(f'Unsupported expression type: {expr!r}')
            class_groups.append(tuple(pruned_group))

        nodes = tuple(self.live_gates[node_id] for node_id in sorted(self.live_gates))
        metadata = dict(self.circuit.metadata)
        metadata.update(
            {
                'pruning_stage': 'naive',
                'original_node_count': self.circuit.node_count,
                'original_layer_count': self.circuit.layer_count,
            }
        )
        return NaivePrunedCircuit(
            input_dim=self.circuit.input_dim,
            nodes=nodes,
            class_groups=tuple(class_groups),
            metadata=metadata,
        )


def naive_prune_circuit(circuit: RegularCircuitIR) -> NaivePrunedCircuit:
    """Stage-1 pruning for extracted compactlogic Boolean circuits.

    This follows the paper’s simple idea:
    - keep only logic reachable from outputs,
    - eliminate constant nodes,
    - eliminate identity/negation nodes by rewiring,
    - count remaining non-trivial gates as BOPs.
    """
    return _NaivePruner(circuit).prune()


def naive_prune_regular_circuit(circuit: RegularCircuitIR) -> NaivePrunedCircuit:
    return naive_prune_circuit(circuit)



def reindex_pruned_circuit(pruned: NaivePrunedCircuit) -> NaivePrunedCircuit:
    """Compact node numbering after stage-1 pruning.

    The pruned circuit semantics are unchanged. We only remap surviving internal
    node ids so they become contiguous after the primary inputs:

    - inputs remain ``0 .. input_dim - 1``
    - kept gates become ``input_dim .. input_dim + bop_count - 1``

    This makes downstream netlists and Verilog smaller and easier to read while
    preserving topological order.
    """
    mapping: dict[int, int] = {}
    next_id = pruned.input_dim
    new_nodes: list[GateNode] = []

    for node in pruned.nodes:
        new_id = next_id
        mapping[node.node_id] = new_id
        next_id += 1

        input_a = node.input_a
        if input_a.signal_id is not None:
            input_a = GateInput(signal_id=mapping.get(input_a.signal_id, input_a.signal_id))
        input_b = node.input_b
        if input_b.signal_id is not None:
            input_b = GateInput(signal_id=mapping.get(input_b.signal_id, input_b.signal_id))
        new_nodes.append(
            GateNode(
                node_id=new_id,
                layer_index=node.layer_index,
                local_index=node.local_index,
                gate_id=node.gate_id,
                input_a=input_a,
                input_b=input_b,
            )
        )

    new_groups: list[tuple[BoolSource, ...]] = []
    for group in pruned.class_groups:
        converted: list[BoolSource] = []
        for source in group:
            if source.signal_id is None:
                converted.append(source)
            else:
                converted.append(
                    BoolSource(
                        signal_id=mapping.get(source.signal_id, source.signal_id),
                        inverted=source.inverted,
                    )
                )
        new_groups.append(tuple(converted))

    metadata = dict(pruned.metadata)
    metadata.update(
        {
            'reindexed': True,
            'reindexed_bop_count': len(new_nodes),
        }
    )
    return NaivePrunedCircuit(
        input_dim=pruned.input_dim,
        nodes=tuple(new_nodes),
        class_groups=tuple(new_groups),
        metadata=metadata,
    )
