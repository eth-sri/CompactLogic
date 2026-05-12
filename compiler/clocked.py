from __future__ import annotations

import argparse
from pathlib import Path

from .extract import extract_circuit_from_checkpoint
from .prune import naive_prune_regular_circuit, reindex_pruned_circuit
from .verilog import _default_pruned_module_name, emit_naive_pruned_verilog


HEADER = """// -----------------------------------------------------------------------------
// Auto-generated clocked wrapper for compactlogic compiled Verilog
// One-sample-per-cycle wrapper with input/output registers.
// -----------------------------------------------------------------------------
"""


ReductionStyle = str


def emit_clocked_wrapper_verilog(
    inner_module_name: str,
    *,
    input_width: int,
    output_width: int,
    wrapper_module_name: str | None = None,
) -> str:
    """Emit a simple synchronous wrapper around a combinational DUT."""
    wrapper_module_name = wrapper_module_name or f"{inner_module_name}_clocked"
    lines = [HEADER.rstrip(), ""]
    lines.extend(
        [
            f"module {wrapper_module_name} (",
            "    input  wire clk,",
            f"    input  wire [{input_width - 1}:0] in_bits,",
            f"    output reg  [{output_width - 1}:0] class_counts_flat",
            ");",
            "",
            f"    reg  [{input_width - 1}:0] in_bits_reg;",
            f"    wire [{output_width - 1}:0] class_counts_flat_comb;",
            "",
            f"    {inner_module_name} dut (",
            "        .in_bits(in_bits_reg),",
            "        .class_counts_flat(class_counts_flat_comb)",
            "    );",
            "",
            "    always @(posedge clk) begin",
            "        in_bits_reg <= in_bits;",
            "        class_counts_flat <= class_counts_flat_comb;",
            "    end",
            "endmodule",
            "",
        ]
    )
    return "\n".join(lines)


def emit_naive_pruned_reindexed_clocked_bundle_from_checkpoint(
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    inner_module_name: str | None = None,
    wrapper_module_name: str | None = None,
    reduction_style: ReductionStyle = "balanced",
) -> str:
    """Emit the preferred current hardware candidate as one Verilog bundle."""
    circuit = extract_circuit_from_checkpoint(checkpoint_path, metadata_path)
    pruned = reindex_pruned_circuit(naive_prune_regular_circuit(circuit))

    inner_module_name = inner_module_name or _default_pruned_module_name(pruned.metadata, reduction_style)
    wrapper_module_name = wrapper_module_name or f"{inner_module_name}_clocked"

    core = emit_naive_pruned_verilog(pruned, module_name=inner_module_name, reduction_style=reduction_style)

    group_size = len(pruned.class_groups[0])
    count_width = max(1, (group_size + 1).bit_length())
    output_width = pruned.num_classes * count_width
    wrapper = emit_clocked_wrapper_verilog(
        inner_module_name,
        input_width=pruned.input_dim,
        output_width=output_width,
        wrapper_module_name=wrapper_module_name,
    )
    return core.rstrip() + "\n\n" + wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a clocked Verilog wrapper around the preferred compiled core.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to ckpt_best_*.pt")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional path to sibling meta_data.json")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the generated bundled .v file")
    parser.add_argument("--inner-module-name", type=str, default=None, help="Optional custom core module name")
    parser.add_argument("--wrapper-module-name", type=str, default=None, help="Optional custom wrapper module name")
    parser.add_argument(
        "--reduction-style",
        choices=("linear", "balanced"),
        default="balanced",
        help="How to implement output population-count reduction inside the core.",
    )
    args = parser.parse_args()

    verilog = emit_naive_pruned_reindexed_clocked_bundle_from_checkpoint(
        args.checkpoint,
        args.metadata,
        inner_module_name=args.inner_module_name,
        wrapper_module_name=args.wrapper_module_name,
        reduction_style=args.reduction_style,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(verilog, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
