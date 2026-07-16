# Coordinate & normal conventions

Verified against `villa/lasagna/tifxyz_format.md` and the `vesuvius` tifxyz API

## tifxyz surface

- A surface is a directory of `x.tif`, `y.tif`, `z.tif` (single-channel float32,
  identical H×W), plus `meta.json` (`format="tifxyz"`, `scale=[sx, sy]`)
- Grid vertex `(row, col)` → world point `P = (X, Y, Z)` where
  `X = x.tif[row, col]`, etc.
- Validity: a vertex is invalid if `Z <= 0` (load-time rule). Optional `mask.tif`
  (channel 0 `< 255` invalidates) further masks vertices. On write, ScrollAnchor
  sets invalid vertices to `(-1, -1, -1)` and writes a `uint8` `mask.tif`
- `scale` is preserved on copy. ScrollAnchor does not resample the grid, so
  `scale` is passed through unchanged

## CT volume

- A CT volume is indexed `vol[z, y, x]` (matches `vesuvius.Volume[z, y, x]`)
- A world point `P = (X, Y, Z)` samples the volume at index `(Z, Y, X)`.
  Sampling uses `scipy.ndimage.map_coordinates` (trilinear by default)
- An in-memory ROI may carry an `origin = (z0, y0, x0)`; world point `P` maps to
  local index `(Z - z0, Y - y0, X - x0)`

## Normals

- Per-vertex normal `= normalize(cross(dP/dcol, dP/drow))` in world (X, Y, Z)
- Signs are made consistent within a surface (flipped into a single hemisphere by
  the mean normal) so signed drift offsets are comparable across the grid
- All offsets, radii, spacings, and windows are in **voxels** (grid vertices for
  window sizes)

## Units summary

| Quantity | Unit |
|---|---|
| `sampling.radius`, `step`, offsets, drift, spacing | voxels |
| `smooth_window`, `switch_smooth_window`, review region sizes | grid vertices |
| confidence, margin, switch_score, evidence, contrast | dimensionless [0, 1] |
