# PHerc combined render diagnostics

The published lightweight core of a completed `analyze-render` run on the
PHercParis4 w110-112 surface render.

This directory holds the machine-readable outputs only: the exported candidate
list, the run metadata, the count summary, and the review contact sheet. The large
numerical arrays and full-resolution overlays from the original run are omitted.
Everything here was produced by a single completed run and copied unmodified.

## Source run

Read from the copied `metadata.json` and `summary.json`:

| Property | Value |
| --- | --- |
| source render shape (rows x cols) | `6270 x 24030` |
| processed shape (rows x cols) | `3135 x 12015` |
| working downsample | `2` |
| full-render factor | `8` |
| total response components | 1152 |
| above-threshold components | 428 |
| exported candidates | 200 |
| suppressed by the export cap | 228 |
| raw score range | `0.133379` to `0.200628` |
| runtime | `410.119` seconds |
| peak resident memory | `3027.0` MB (about 3.0 GB) |

The 200 exported candidates are the ranked subset, ordered by strictly decreasing
raw detector score. The detector response, the raw detector score, the exported
rank, and the component ID are four separate things and are kept separate here: an
exported rank is a position in this list, while a component ID identifies a
connected component of the detector response and is unrelated to rank.

## Diagnostics enabled

This run carried **two separate and independent diagnostic systems**. Both are
reviewer aids attached to each exported candidate. Neither changed the detector.

### 1. Normalized tifxyz valid-surface diagnostics

- normalized tifxyz shape: `627 x 2403`
- validity convention: x, y and z all finite and none equal to the exact `-1`
  sentinel. `mask.tif` is not consulted. Zero is a legitimate coordinate and stays
  valid
- valid fraction of the tifxyz raster: `0.866221`
- raster correspondence: an exact positive integer scale between the tifxyz grid
  and the analysis rasters, verified per axis, with no resampling of the tifxyz grid
- orientation: identity is **assumed**, not proven. The stored
  `alignment_assumptions` block records `independently_verified: false`. Only the
  shared integer raster scale is verified; the origin, orientation and extent
  correspondence has not been independently confirmed from dataset metadata
- per-candidate fields: `invalid_overlap_fraction`, `touches_invalid_boundary`
- **effect on scoring: none.** The stored `effect_on_scoring` field reads
  `"none; diagnostics only in this revision"`

### 2. Render-derived background diagnostics

Derived from the processed render itself, with no surface geometry involved:

- grayscale threshold: `32` (8-bit)
- minimum retained component size: `5000` processed pixels
- connectivity: 4-neighbor
- only **edge-connected** near-black components are retained. Of 11638 near-black
  components, 11 were border-connected and 1 was retained after the size filter
- background fraction of the processed render: `0.309601`
- per-candidate fields: `render_background_overlap_fraction`,
  `render_background_distance_px`, `touches_render_background`
- **effect on scoring: none.** The stored `effect_on_scoring` field reads
  `"none; diagnostics only"`

### These two are not the same measurement

Render-derived background is **not** equivalent to tifxyz-invalid surface. They
describe different things and can disagree over large areas: one is visible
near-black pixels in a 2D render, the other is missing surface coordinates in the
tifxyz grid. Both are reviewer diagnostics. Neither confirms a sheet skip and
neither confirms a reconstruction error. The run's own metadata records this
distinction as an explicit limitation.

## Key observations

Verified directly from the copied `regions.json`:

- five visually obvious black-background cases, at original exported ranks 5, 6, 8,
  10 and 16, all carried positive component-level render-background overlap
  (`0.929032`, `0.835938`, `0.600000`, `0.240385` and `0.654206`)
- original rank 18 also carried high render-background overlap (`0.915670`)
- original ranks 7 and 11 carried zero render-background overlap and served as
  interior controls, sitting `661.653` and `981.041` processed pixels away from the
  retained background component
- tifxyz boundary distance did not explain the visible black render wedges. Every
  one of the six background-heavy ranks above recorded
  `invalid_overlap_fraction: 0.0` and `touches_invalid_boundary: false`. Across the
  whole exported subset only 1 of 200 candidates showed any tifxyz-invalid overlap,
  against 67 of 200 with nonzero render-background overlap
- that gap is what motivated the separate ranking-policy audit, which asked whether
  a background-aware secondary review order would put a better set of candidates in
  front of a reviewer first
- the same exported regions were later reused as the input for candidate-to-tifxyz
  mapping of ranks 7 and 11

None of these candidates is a confirmed false positive. High render-background
overlap means a candidate sits largely on near-black render area, which is a reason
for a reviewer to look at it differently, not an adjudication.

## Published files

- [regions.json](regions.json) - the 200 exported candidates with component ID, raw
  score, direction, bounding boxes and centroids in both source-JPG and processed
  coordinates, plus both sets of diagnostic fields
- [metadata.json](metadata.json) - shapes, scales, all detector parameters, runtime,
  peak memory, region counts, exported score range, and the full configuration and
  limitation blocks for both diagnostic systems
- [summary.json](summary.json) - the count funnel and the exported score range
- [top_candidates.png](top_candidates.png) - contact sheet of the top-ranked
  candidate crops

Deliberately not stored in this repository:

- the source JPG render
- the tifxyz arrays
- `diagnostics.npz`, the large numerical seam and anomaly arrays
- the full-resolution overlay and the review PDF

The two JSON files preserve the exact exported candidates and the exact diagnostic
fields that the published downstream artifacts consume, so those results remain
inspectable without the large inputs.

The contact sheet is a 2D review artifact. It is not a CT overlay and it is not CT
validation.

## Downstream artifacts

Two published artifacts were built from this run:

- [../pherc-ranking-policy-audit/README.md](../pherc-ranking-policy-audit/README.md)
  consumed `regions.json`, `metadata.json` and `summary.json` from this run, using
  the `render_background_overlap_fraction` field to compare hypothetical
  review-order policies. It also read the external source JPG once for its
  contact-sheet crops. No production ranking change was adopted
- [../pherc-candidate-tifxyz-mapping/README.md](../pherc-candidate-tifxyz-mapping/README.md)
  consumed `regions.json` and `metadata.json` from this run to map exported ranks 7
  and 11 onto normalized tifxyz raster cells. It additionally required the external
  tifxyz arrays
- [../pherc-alignment-audit/README.md](../pherc-alignment-audit/README.md)
  investigated exactly why the tifxyz-invalid diagnostic did not account for all the
  black render wedges seen here. It compared the two masks under four simple
  orientation hypotheses and found the gap is not a misalignment: identity gave the
  strongest agreement of the four, and tifxyz-invalid surface turned out to be a
  subset of the broader render-derived background, which covers interior wedges that
  still carry valid surface coordinates. It read the candidate list and the external
  render and tifxyz inputs

Both downstream artifacts pass the raw detector scores and the exported order
through unchanged.

## Attribution

Source render derived from Vesuvius Challenge open data (PHercParis4 segment
20260623163339-w110-112). This artifact does not imply endorsement by the Vesuvius
Challenge.

- Author: Olga Ivanova
- Repository: https://github.com/olgaiv39/scroll-anchor
