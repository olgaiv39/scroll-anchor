# CLAUDE.md

## Project
ScrollAnchor: conservative, read-only, CPU-only, ground-truth-free surface-label
diagnostics for volumetric papyrus CT. It flags normal-direction **drift** and
neighbouring-**sheet-switch** errors in an approximate `tifxyz` surface; it does
not move labels (correction proposals exist but are off by default).

## Author
Olga Ivanova — ivolga.vak@gmail.com. Preserve this as the sole author and
copyright holder in all metadata (`pyproject.toml`, `LICENSE`, `CITATION.cff`,
`src/scroll_anchor/__init__.py`). Do not replace with generic wording. Do not
expose the email in runtime output or generated artifacts.

## Layout
- `src/scroll_anchor/` — package (tifxyz I/O, volume ROI, normals, sampling,
  diagnostics, report, CLI, synthetic benchmark).
- `nrrd_io.py`, `realcube.py`, `previews.py` — real-cube benchmark modules.
- `scripts/run_real_cube_benchmark.py` — real-cube workflow entry point.
- `tests/` — unit tests (no large real data).
- `results/` — small metadata/metrics/previews only; arrays and NRRD are gitignored.

## Reference material: villa
A local checkout of `ScrollPrize/villa` sits alongside the repo at
`../villa` (sibling of `scroll-anchor`). It is **read-only reference only**: never
modify, format, copy large parts, stage, or commit it.

## Internal conventions
- CT volume is indexed `[z, y, x]`; a world point `P = (X, Y, Z)` samples index
  `(Z, Y, X)`. tifxyz surfaces store world `(X, Y, Z)` per grid vertex.
- Instance-label NRRD cubes store axes in `(z, y, x)` order (folder `z_y_x`,
  `space origin` matches). `nrrd_io.read_nrrd` resolves this to internal `[z,y,x]`
  and fails loudly on non-axis-aligned or ambiguous metadata — never transposes
  silently.
- Offsets, radii, spacings are in **voxels**; window sizes are in grid vertices.

## Commands
- Tests: `python -m pytest -q`
- Synthetic benchmark: `scroll-anchor benchmark --output results/bench --seed 0`
- Real-cube benchmark: `python scripts/run_real_cube_benchmark.py --output results/real_cube_02256_02512_04816`
  (add `--offline` to use already-downloaded NRRD in `data/real_cube/`).
- Benchmark extras: `pip install -e ".[benchmark]"` (pynrrd, matplotlib).

## Git
Repository is private. Do NOT commit, push, tag, release, `git add`, amend, or
change visibility without an explicit instruction. Inspecting `git diff`/`status`
is fine. The user reviews and commits.

## Style
Concise comments and docstrings; explain only geometry, axis conversion, or
non-obvious safety decisions. No speculative abstractions or broad refactors.

## Honesty
Keep validation claims honest. The real-cube benchmark uses real CT + real sheet
geometry with **controlled, injected** corruptions — it is not validation on
naturally occurring annotation errors. Report negative/mixed results as-is.
