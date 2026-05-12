# third_party/convlogic

Minimal vendored ConvLogic adapter used by this repository's compiler and simulation stack.

## Scope

This directory intentionally contains only the small subset of ConvLogic functionality needed for:

1. loading supported ConvLogic checkpoints,
2. reconstructing their native runtime for semantic evaluation,
3. extracting them into the common circuit IR used by this repository, and
4. generating compiler-style reports through the shared simulation pipeline.

It is **not** intended to vendor the full upstream training codebase.

## Contents

- `config.py`
  - infer the supported ConvLogic checkpoint configuration
- `evaluate.py`
  - reconstruct and run the native ConvLogic runtime on MNIST/CIFAR checkpoints
- `extract.py`
  - convert supported ConvLogic checkpoints into this repository's `RegularCircuitIR`
- `paper_stats.py`
  - generate compiler-style reports for ConvLogic checkpoints

## Usage

Generate a ConvLogic compiler report with:

```bash
python -m third_party.convlogic.paper_stats \
  --checkpoint <path_to_convlogic_checkpoint.ckpt>
```

This writes the report files next to the checkpoint by default:

- `convlogic_compiler_paper_stats.json`
- `convlogic_compiler_paper_stats.md`
- emitted balanced clocked Verilog

## Notes

- The ConvLogic report path uses the same shared simulation/report infrastructure as CompactLogic where possible.
- Standard reports require full-test semantic validation by default.
- Partial semantic validation is available only as an explicit debug option.
