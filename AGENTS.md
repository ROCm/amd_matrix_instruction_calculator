# AI Rules

This file provides guidance to AI coding agents working with code in this repository.

Single-file Python CLI (`matrix_calculator.py`) that maps matrix elements ↔ registers/lanes for AMD MFMA/SMFMAC (CDNA1–3) and WMMA/SWMMAC (RDNA3–4) instructions. `README.md` is the spec, with expected output for every option.

## Commands

```bash
pip install -r requirements.txt -r requirements-test.txt pylint

./matrix_calculator.py -a cdna3 -L                                 # list instructions
./matrix_calculator.py -a cdna3 -i v_mfma_f32_32x32x8_f16 -d       # instruction details
./matrix_calculator.py -a rdna3 -i v_wmma_f32_16x16x16_f16 -A -R   # A-matrix register layout

pylint matrix_calculator.py test/delta_test.py                     # must stay 10.00/10
```

## Testing

There are no unit tests and no golden output. `test/delta_test.py` runs the tool over a huge set of valid and invalid command lines and writes all output to one file (about 30 MB, 1–2 min). Run it before and after a change, then diff:

```bash
./test/delta_test.py -o /tmp/before.txt   # on the base commit
./test/delta_test.py -o /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
```

The tester reads the architecture list from the tool's `--help` output (the "following architectures" line and the "Alternately:" lines). Keep that format, or update `get_architectures`/`get_alt_architectures`. Per-arch wave sizes and lane limits are hardcoded in `get_supported_wave_sizes`/`get_max_lane_num`; update them when adding an architecture.

## Architecture

- **Data tables** (first ~2900 lines): `dict_isas` (alias → arch key), `dict_math_types`, and `dict_insts[arch][mnemonic]` (`MatrixInstruction` TypedDicts with dimensions, cycles, and modifier-support flags). Adding an instruction or alias mostly means editing these, plus the README.
- **`parse_and_run()`**: all argument validation (errors go to stderr and return `-2`), then it picks the calculator class via `is_gfx9_arch`/`is_gfx11_arch`/`is_gfx12_arch`.
- **Calculator classes**: `InstCalc` (abstract base with the generic table building and output formatting) is subclassed by `InstCalcGfx9` (CDNA), `InstCalcGfx11` (RDNA3), and `InstCalcGfx12` (RDNA4). The core per-arch hook is `_get_reg_lanes()`. `InstCalcGfx12`'s docstring says it is a child of Gfx11, but it inherits from `InstCalc`.

## Conventions

- Must run on old Python 3. Use `typing.Dict`/`List`/`Tuple`, not `dict[...]`/`tuple[...]` (this broke Python < 3.9 before). `TypedDict` falls back to `typing_extensions`.
- Bump `VERSION` in `matrix_calculator.py` on every code change (and in `test/delta_test.py` when the tester changes). Keep `README.md` in sync.
