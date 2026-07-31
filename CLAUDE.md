# CLAUDE.md

## Project
ScrollAnchor: conservative, read-only, CPU-only, ground-truth-free surface-label
diagnostics for volumetric papyrus CT. It flags normal-direction **drift** and
neighbouring-**sheet-switch** errors in an approximate `tifxyz` surface; it does
not move labels (correction proposals exist but are off by default).

## Public repository
This repository is public. Never add secrets, credentials, API tokens, private or
embargoed datasets, machine-specific absolute paths, or unpublished third-party
material. Use relative paths in code, configs, docs, and committed artifacts.

## Author
Olga Ivanova - ivolga.vak@gmail.com. Preserve this as the sole author and
copyright holder in all metadata (`pyproject.toml`, `LICENSE`, `CITATION.cff`,
`src/scroll_anchor/__init__.py`). Do not replace with generic wording. Do not
expose the email in runtime output or generated artifacts.

## Layout
- `src/scroll_anchor/` - package: tifxyz I/O, volume ROI, normals, sampling,
  diagnostics, report, CLI, synthetic benchmark, plus `nrrd_io.py`, `realcube.py`
  and `previews.py` for the real-cube benchmark.
- `scripts/run_real_cube_benchmark.py` - real-cube workflow entry point.
- `configs/` - YAML configuration.
- `tests/` - unit tests; no large real data.
- `results/` - small metadata, metrics and previews only. Arrays, NRRD cubes and
  source renders stay gitignored.

## Workflows
Core workflow (3D, the primary contribution): `scroll-anchor analyze` takes a
`tifxyz` surface plus a CT volume or ROI and produces localized drift and switch
diagnostics from through-thickness CT evidence and surface-normal geometry.

Exploratory 2D workflow (separate and secondary): `analyze-render` exports review
candidates from a surface render, `render-report` rebuilds the review PDF from an
existing result directory, and `map-render-candidates` maps candidates onto
normalized tifxyz cells under an explicitly acknowledged identity-orientation
assumption. This workflow prioritizes review; it is not volumetric detection.

## Evidence status
Keep these three levels distinct and never merge them:
- synthetic benchmark - precise detection under **controlled synthetic**
  corruptions.
- real-cube benchmark - real CT and real sheet geometry with **controlled,
  injected** corruptions. It supports the conservative safety concept; it is not
  validation on naturally occurring annotation errors.
- naturally occurring real annotation failures - **not yet evaluated** against
  full ground truth.

PHerc render candidates and local CT crops are supporting evidence for human
review. They are not confirmed sheet skips, confirmed reconstruction failures,
confirmed false positives, or full volumetric validation. Do not claim proven
orientation, solved registration, or verified coordinate provenance. Report
negative and mixed results as-is.

## Internal conventions
- CT volume is indexed `[z, y, x]`; a world point `P = (X, Y, Z)` samples index
  `(Z, Y, X)`. tifxyz surfaces store world `(X, Y, Z)` per grid vertex.
- Instance-label NRRD cubes store axes in `(z, y, x)` order (folder `z_y_x`,
  `space origin` matches). `nrrd_io.read_nrrd` resolves this to internal `[z,y,x]`
  and fails loudly on non-axis-aligned or ambiguous metadata - never transposes
  silently.
- Offsets, radii, spacings are in **voxels**; window sizes are in grid vertices.

## Commands
- Tests: `python -m pytest -q`
- Synthetic benchmark: `scroll-anchor benchmark --output results/bench --seed 0`
- Real-cube benchmark: `python scripts/run_real_cube_benchmark.py --output results/real_cube_02256_02512_04816`
  (add `--offline` to use already-downloaded NRRD in `data/real_cube/`)
- Extras: `".[remote]"` (zarr, fsspec), `".[benchmark]"` (pynrrd, matplotlib),
  `".[render]"` (Pillow, matplotlib), `".[dev]"` (pytest)

## Reference material: villa
`ScrollPrize/villa` may exist as a sibling checkout at `../villa`. Use it only if
that checkout is already present; do not assume it exists and do not fetch it. It
is **read-only reference only**: never modify, format, copy large parts, stage, or
commit it.

## Git
Do NOT commit, push, tag, release, `git add`, amend, or change repository settings
without an explicit instruction. Inspecting `git diff` and `git status` is fine.
The user reviews and commits.

## Style
Concise comments and docstrings; explain only geometry, axis conversion, or
non-obvious safety decisions. No speculative abstractions or broad refactors.
