# PHerc candidate-to-tifxyz mapping

Render-to-surface-coordinate mapping for two exported candidates from the
PHercParis4 w110-112 exploratory render run

This is a **coordinate mapping artifact**. It is not CT validation

## Purpose

`map-render-candidates` is the reproducible bridge between an already exported 2D
render candidate and the normalized `tifxyz` surface grid. For each selected
candidate it walks a fixed chain:

1. selection by exported rank or component ID
2. source-render centroid (`centroid_rowcol_jpg`)
3. processed-render centroid (`centroid_rowcol_processed`)
4. normalized `tifxyz` raster cell (`tifxyz_row`, `tifxyz_col`)
5. the values stored at that cell in `x.tif`, `y.tif` and `z.tif`

The command reads `regions.json`, `metadata.json` and the three coordinate TIFFs.
It writes only its own output directory and never touches the `analyze-render`
run it reads

It does **not**:

- rerun candidate detection
- modify scores or ranking
- infer CT NumPy axis order
- load a CT volume
- confirm a sheet skip
- confirm a surface reconstruction error

Six quantities are kept strictly separate throughout:

| Quantity | Meaning |
| --- | --- |
| exported rank | 1-based position in the ranked subset in `regions.json` |
| component ID | connected-component identifier; unrelated to rank |
| raw detector score | `score` produced by `analyze-render`, passed through unchanged |
| tifxyz raster cell | integer `(row, col)` in the normalized surface grid |
| x/y/z surface coordinates | raw values read from the three coordinate TIFFs |
| CT array indices | not produced by this command |

## Reusable CLI

```bash
scroll-anchor map-render-candidates \
  --regions PATH/regions.json \
  --metadata PATH/metadata.json \
  --tifxyz-dir PATH/tifxyz_directory \
  --rank 7 \
  --rank 11 \
  --render PATH/render.jpg \
  --assume-identity-orientation \
  --output PATH/output_directory
```

- `--rank` and `--component-id` are both repeatable and may be mixed. At least one
  is required
- rank is never treated as a component ID. They are separate namespaces and the
  command never substitutes one for the other
- duplicate selections are deduplicated, so naming the same candidate by rank and
  by component ID yields one entry
- `--assume-identity-orientation` is mandatory. Orientation is never assumed
  silently; without the flag the command refuses to run and explains why
- incompatible raster dimensions fail. The tifxyz grid must divide both the source
  and processed render shapes exactly, checked independently per axis
- no interpolation, clamping, resampling, or nearest-valid fallback is used. An
  out-of-bounds cell is an error, and an invalid cell is reported as invalid rather
  than replaced by a nearby valid one
- `--render` is optional and only controls whether the preview PNG is written

## Mapping convention

Aligned half-open containment:

```
tifxyz_row = floor(source_jpg_row / source_to_tifxyz_row_scale)
tifxyz_col = floor(source_jpg_col / source_to_tifxyz_col_scale)
```

- scale factors are derived from the input shapes recorded in `metadata.json` and
  the shape of the loaded tifxyz rasters. They are not hardcoded
- they must be exact positive integers, verified separately for rows and columns
- the same cell is derived a second time from the processed centroid and the
  processed-to-tifxyz scale. The two paths must resolve to the same cell or the
  candidate fails
- each tifxyz cell covers exactly one `row_scale x col_scale` block of source-JPG
  pixels, half-open on the high side
- the result identifies the tifxyz cell **containing** the candidate centroid
- it is **not** a subpixel surface position

A cell counts as valid only when x, y and z are all finite and none of them equals
the exact `-1` invalid sentinel. Zero is a legitimate coordinate and stays valid

## PHerc result

Two candidates were mapped from the exported ranked subset of 200 candidates

| Exported rank | Component ID | Direction | Tifxyz row | Tifxyz col | Valid | x | y | z |
|---:|---:|---|---:|---:|---|---:|---:|---:|
| 7 | 566 | vertical | 350 | 2190 | true | 5404.456 | 5874.218 | 10886.725 |
| 11 | 690 | horizontal | 399 | 2066 | true | 3683.824 | 4615.540 | 10383.773 |

Shapes and derived scales:

- source render shape: `6270 x 24030`
- processed render shape: `3135 x 12015`
- normalized tifxyz shape: `627 x 2403`
- source-to-tifxyz scale: `10 x 10`
- processed-to-tifxyz scale: `5 x 5`

Checks that passed for both candidates:

- the source-derived and processed-derived cells agreed
- both selected cells were valid under the finite and exact `-1` sentinel rule

The raw detector scores and exported ranks are carried through unchanged. Neither
candidate is a confirmed sheet skip, a confirmed reconstruction error, or a
confirmed false positive. They remain exploratory 2D visual anomalies that now
have a surface coordinate attached to them

## Orientation and CT limitations

- identity render-to-tifxyz orientation is **explicitly assumed**, on request, via
  `--assume-identity-orientation`. The tifxyz raster is treated as sharing the
  render's origin, row/column orientation and extent, with no flip, transpose,
  crop or offset
- that correspondence is **not** encoded in the supplied metadata, so this command
  cannot and does not verify it. Running the command does not prove it. If the
  assumption is wrong, every mapped cell here is wrong
- x/y/z values are read in file-name order from `x.tif`, `y.tif` and `z.tif`, with
  no unit conversion, no scaling and no axis reordering
- they are **not yet converted to CT NumPy array indices**
- CT volume axis order, bounds, voxel convention, and crop provenance remain
  unverified
- an array may use `(z, y, x)` ordering, but this artifact does not assume it
- exact integer scale division confirms only that the two rasters are dimensionally
  compatible. It is not evidence of correct orientation

A separate offline audit does provide supporting evidence. It compared the
`tifxyz`-invalid mask against the render-derived background under four simple
orientation hypotheses and found identity strongest on every metric (mean IoU
`0.4348`, against `0.3435` for a horizontal flip, `0.1995` vertical and `0.2008`
both):
[`../pherc-alignment-audit/README.md`](../pherc-alignment-audit/README.md)

That is supporting evidence, not an official metadata guarantee. Only four rigid
hypotheses were tested, with no registration search, and the audit records its
overall status as mixed. Identity therefore remains an **explicit assumption** in
the CLI: `map-render-candidates` still requires `--assume-identity-orientation` and
that requirement is unchanged

The next stage is local CT validation at these coordinates, not another 2D
classification step. Until that is done, nothing here validates 3D geometry: a flat
render carries no through-thickness CT evidence and no surface-normal geometry

## Artifacts

- [candidate_tifxyz_mapping.json](candidate_tifxyz_mapping.json) - full record:
  input paths, all three shapes, both derived scales with their derivation rule,
  the orientation assumption, the pixel-containment and invalid-cell conventions,
  the coordinate interpretation warning, and per candidate the rank, component ID,
  score, direction, source and processed centroids and bounding boxes, both cell
  derivations with their agreement flag, validity, and x/y/z
- [candidate_tifxyz_mapping.csv](candidate_tifxyz_mapping.csv) - one flat row per
  candidate for quick inspection, ordered by exported rank then component ID
- [candidate_tifxyz_mapping.png](candidate_tifxyz_mapping.png) - render-to-tifxyz
  **mapping preview**, not a CT overlay. One panel per candidate showing the
  candidate bounding box in yellow, the source-JPG footprint of the mapped tifxyz
  cell in cyan, and the source centroid as a red cross, labelled with exported
  rank, component ID, raw score, tifxyz row and column, validity, and x/y/z

The source render JPG and the tifxyz arrays are **not stored in this repository**.
The committed JSON, CSV and PNG preserve the completed result without them

## Provenance

The candidate regions and metadata this mapping consumed are published. The
`regions.json` and `metadata.json` inputs come from the combined-diagnostics run at
[`../pherc-render-combined-diagnostics/`](../pherc-render-combined-diagnostics/README.md),
which supplies the exported ranks, component IDs and raw scores used above

Rerunning the mapping still requires the external normalized tifxyz arrays, since
`x.tif`, `y.tif` and `z.tif` are what the coordinates are read from and they are not
committed here. Regenerating the optional preview PNG additionally requires the
external source JPG

## Subsequent CT follow-up

A later registration-supported transformed-coordinate interpretation was used to
extract local CT evidence for exported ranks 7 and 11. That follow-up does not
retroactively change the guarantees of the mapping command and does not verify
global coordinate provenance. Mapping itself still produces no native CT indices,
identity orientation remains an explicit assumption, exported rank and component ID
remain distinct, and this mapping artifact alone is not CT validation. The evidence
is documented in
[`../pherc-ct-local-evidence/README.md`](../pherc-ct-local-evidence/README.md)

## Attribution

Source render derived from Vesuvius Challenge open data (PHercParis4 segment
20260623163339-w110-112). This artifact does not imply endorsement by the Vesuvius
Challenge

- Author: Olga Ivanova
- Repository: https://github.com/olgaiv39/scroll-anchor
