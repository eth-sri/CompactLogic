from __future__ import annotations

import torch


GATE_NAMES: tuple[str, ...] = (
    "CONST0",
    "AND",
    "A_AND_NOT_B",
    "A",
    "NOT_A_AND_B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "NOT_B",
    "B_IMPLIES_A",
    "NOT_A",
    "A_IMPLIES_B",
    "NAND",
    "CONST1",
)

CONST_GATE_IDS: frozenset[int] = frozenset({0, 15})
TRIVIAL_GATE_IDS: frozenset[int] = frozenset({3, 5, 10, 12})


def truth_eval_scalar(gate_id: int, a: int, b: int) -> int:
    a = int(bool(a))
    b = int(bool(b))
    if gate_id == 0:
        return 0
    if gate_id == 1:
        return a & b
    if gate_id == 2:
        return a & (1 - b)
    if gate_id == 3:
        return a
    if gate_id == 4:
        return (1 - a) & b
    if gate_id == 5:
        return b
    if gate_id == 6:
        return a ^ b
    if gate_id == 7:
        return a | b
    if gate_id == 8:
        return 1 - (a | b)
    if gate_id == 9:
        return 1 - (a ^ b)
    if gate_id == 10:
        return 1 - b
    if gate_id == 11:
        return (1 - b) | a
    if gate_id == 12:
        return 1 - a
    if gate_id == 13:
        return (1 - a) | b
    if gate_id == 14:
        return 1 - (a & b)
    if gate_id == 15:
        return 1
    raise ValueError(f"Unsupported gate id: {gate_id}")


def eval_gate_torch(gate_id: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if gate_id == 0:
        return torch.zeros_like(a)
    if gate_id == 1:
        return a & b
    if gate_id == 2:
        return a & (~b)
    if gate_id == 3:
        return a
    if gate_id == 4:
        return (~a) & b
    if gate_id == 5:
        return b
    if gate_id == 6:
        return a ^ b
    if gate_id == 7:
        return a | b
    if gate_id == 8:
        return ~(a | b)
    if gate_id == 9:
        return ~(a ^ b)
    if gate_id == 10:
        return ~b
    if gate_id == 11:
        return (~b) | a
    if gate_id == 12:
        return ~a
    if gate_id == 13:
        return (~a) | b
    if gate_id == 14:
        return ~(a & b)
    if gate_id == 15:
        return torch.ones_like(a)
    raise ValueError(f"Unsupported gate id: {gate_id}")


def verilog_gate_expr(gate_id: int, a: str, b: str) -> str:
    if gate_id == 0:
        return "1'b0"
    if gate_id == 1:
        return f"({a} & {b})"
    if gate_id == 2:
        return f"({a} & ~{b})"
    if gate_id == 3:
        return a
    if gate_id == 4:
        return f"(~{a} & {b})"
    if gate_id == 5:
        return b
    if gate_id == 6:
        return f"({a} ^ {b})"
    if gate_id == 7:
        return f"({a} | {b})"
    if gate_id == 8:
        return f"~({a} | {b})"
    if gate_id == 9:
        return f"~({a} ^ {b})"
    if gate_id == 10:
        return f"~{b}"
    if gate_id == 11:
        return f"(~{b} | {a})"
    if gate_id == 12:
        return f"~{a}"
    if gate_id == 13:
        return f"(~{a} | {b})"
    if gate_id == 14:
        return f"~({a} & {b})"
    if gate_id == 15:
        return "1'b1"
    raise ValueError(f"Unsupported gate id: {gate_id}")
