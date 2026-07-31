# PHerc render-to-tifxyz alignment audit

Offline audit asking whether the normalized `tifxyz` validity raster and the
PHercParis4 w110-112 render-derived edge-connected near-black background are
consistent under simple orientation hypotheses

## Result summary

- identity was the strongest of the four tested simple orientation hypotheses
- this does not establish the official orientation and does not solve registration

## Purpose

The combined-diagnostics run carried two independent diagnostic masks that did not
agree: `tifxyz`-invalid surface flagged almost nothing, while render-derived
background covered large visible black wedges. This audit asked whether that gap
could be an orientation error, by measuring mask agreement under four orientation
hypotheses:

- identity
- horizontal flip
- vertical flip
- horizontal plus vertical flip

The audit did **not**:

- modify candidate detection
- modify scores or ranking
- prove an official metadata orientation convention
- establish CT array indexing
- validate a sheet skip
- validate a reconstruction error

Detection was not rerun and scores were not recomputed. The audit's own summary
records `detection_rerun: false` and `scores_recomputed: false`, and reuses the
existing 200 exported candidates with score range `0.133379` to `0.200628`

## Inputs and method

All values below are read from the copied artifacts

| Property | Value |
| --- | --- |
| source render shape (rows x cols) | `6270 x 24030` |
| processed raster shape | `3135 x 12015` |
| normalized tifxyz shape | `627 x 2403` |
| JPG to tifxyz scale | `10.0 x 10.0`, exact integer |
| processed to tifxyz scale | `5.0 x 5.0`, exact integer |
| grayscale thresholds evaluated | `1, 4, 8, 12, 16, 24, 32` |
| minimum retained component size | `5000` processed pixels |
| connectivity | 4-neighbor |
| background rule | edge-connected near-black components only |
| orientation hypotheses | identity, hflip, vflip, hvflip |
| transpose tested | no |
| representative threshold | `32` |
| candidates examined | 200 reused, 8 inspected in detail |

Comparison measurements, computed per threshold and per orientation between the
projected `tifxyz`-invalid mask and the render-derived background mask:

- intersection over union (IoU)
- render-background precision and recall with respect to `tifxyz`-invalid
- median and p90 contour distance in both directions, in processed pixels

Boundary sets are deterministically stride-subsampled to at most 200000 pixels per
set, with the stride recorded per row

Transpose was not tested, and the artifact records why: the raster dimensions
(`627x2403` against `3135x12015`) make it incompatible. This is a statement about
dimensional compatibility, not about orientation

## Main result

Identity produced the strongest agreement among the four tested hypotheses. Mean
IoU across the seven thresholds:

| Orientation | Mean IoU |
| --- | ---: |
| identity | `0.434773` |
| hflip | `0.343546` |
| vflip | `0.199496` |
| hvflip | `0.200847` |

All five recorded metrics selected identity, and the audit records
`metrics_agree_on_one_orientation: true` with `best_orientation_overall: identity`.
The recorded `mean_iou_uplift_over_identity` is `0.0`, meaning no tested
alternative beat it

At the representative threshold `32`:

| Orientation | IoU | Precision | Recall | Median bg to tifxyz (px) | Median tifxyz to bg (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | `0.432103` | `0.432103` | `1.0` | `257.0` | `4.0` |
| hflip | `0.343473` | `0.366132` | `0.847326` | `311.0` | `66.008` |
| vflip | `0.199852` | `0.238537` | `0.552036` | `340.777` | `143.003` |
| hvflip | `0.200815` | `0.239494` | `0.554252` | `366.947` | `158.003` |

### Containment, not equivalence

- `tifxyz`-invalid pixels were contained within the broader render-derived
  background over the tested thresholds. Under identity, recall with respect to
  `tifxyz`-invalid is `1.0` at thresholds 4, 8, 12, 16, 24 and 32, and `0.999985`
  at threshold 1
- the `tifxyz` contour sat close to the render-background contour, while the
  reverse distance was much larger: median `4.0` px from `tifxyz` to background
  against `257.0` px from background to `tifxyz` at threshold 32
- the render-derived background covered substantially more area than
  `tifxyz`-invalid surface. At threshold 32 the background fraction is
  `0.30960069` (about `0.3096`) against a `tifxyz`-invalid fraction of `0.133779`
  (about `0.1338`)
- IoU therefore stays near `0.43` even in the best case, because one mask is
  roughly a subset of a mask more than twice its size

That asymmetry supports **different validity semantics** rather than treating the
two masks as equivalent. `tifxyz`-invalid marks the outer margin where the surface
parameterization has no coordinates; the render-derived background additionally
covers interior near-black wedges that carry valid surface coordinates. The
`mask_alignment_overview.png` figure shows this directly: the projected
`tifxyz`-invalid contour traces the outer sheet boundary, while background contours
also enclose interior wedges

## Interpretation

Careful wording matters here:

- the audit provides **evidence** that identity is the best of the four tested
  simple orientation hypotheses
- it does **not** prove that identity is the official or universally correct
  convention. Only four rigid hypotheses were compared; no registration search was
  performed
- exact raster divisibility alone is **not** orientation evidence. That the tifxyz
  grid divides the render shape exactly establishes dimensional compatibility only
- candidate mapping therefore still requires the explicit
  `--assume-identity-orientation` acknowledgement in `map-render-candidates`. This
  audit does not remove that requirement and does not change that CLI behaviour
- render-derived background is not equivalent to `tifxyz`-invalid surface

The audit's own recorded verdict is deliberately reserved:
`conclusion_status: "mixed_or_inconclusive"`, with the rationale that the evidence
"does not satisfy any single interpretation cleanly". The orientation comparison is
one clear signal inside an otherwise mixed result, and the artifact's own note
states that "no single metric is sufficient to declare alignment"

## Candidate observations

Eight of the 200 exported candidates were inspected in detail at threshold 32.
Exported rank and component ID are distinct: rank is a position in the ranked
subset, component ID identifies a connected component of the detector response

| Exported rank | Component ID | Group | Score | Render-bg overlap | Render-bg distance (px) | Touches bg | tifxyz-invalid overlap | tifxyz boundary distance (px) |
| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 5 | 215 | suspected black edge | `0.181901` | `0.949772` | `0.0` | yes | `0.0` | `206.342` |
| 6 | 1064 | suspected black edge | `0.180722` | `0.828302` | `0.0` | yes | `0.0` | `274.536` |
| 7 | 566 | interior control | `0.178003` | `0.0` | `660.734` | no | `0.0` | `787.610` |
| 8 | 795 | suspected black edge | `0.172064` | `0.674419` | `0.0` | yes | `0.0` | `117.098` |
| 10 | 965 | suspected black edge | `0.170168` | `0.219355` | `0.0` | yes | `0.0` | `647.000` |
| 11 | 690 | interior control | `0.168743` | `0.0` | `980.340` | no | `0.0` | `984.403` |
| 15 | 638 | suspected black edge | `0.166563` | `0.0` | `53.000` | no | `0.0` | `656.880` |
| 16 | 78 | suspected black edge | `0.166419` | `0.635593` | `0.0` | yes | `0.0` | `296.978` |

- ranks 5, 6, 8, 10 and 16 showed render-background intersection in the audit, at
  all seven thresholds each
- ranks 7 and 11 remained interior candidates, with zero background overlap and no
  threshold at which they touched the background
- rank 15 remained ambiguous. It was grouped as a suspected black-edge case, yet
  recorded zero background overlap while sitting only `53.0` px away, so it is
  near the background without intersecting it. The audit records
  `suspected_black_edge_consistently_close_to_render_boundary: false`, and rank 15
  is the reason
- every one of the eight recorded `tifxyz_invalid_overlap_fraction: 0.0` and
  `tifxyz_touches_invalid_boundary: false`, with `tifxyz` boundary distances from
  `117` to `984` px. The `tifxyz` diagnostic did not account for the black wedges
- all eight were stable across thresholds

These are **diagnostic observations, not confirmed labels**. No candidate here is a
confirmed sheet skip, a confirmed reconstruction error, or a confirmed false
positive

Two caveats when comparing these numbers to the combined-diagnostics run:

- this audit approximates a candidate footprint by its **bounding box**, because
  `regions.json` stores a bbox and size but not the component pixel mask. Overlap
  fractions here are therefore bbox-proxy values and differ slightly from the
  component-level `render_background_overlap_fraction` published in
  [`../pherc-render-combined-diagnostics/`](../pherc-render-combined-diagnostics/README.md).
  The two agree on which ranks intersect the background and which do not
- the audit also reports an internal projection reconstruction check with
  `faithful: false` and 24 deviations above its tolerance, so the projected
  `tifxyz` mask is an approximation rather than an exact reconstruction

## Artifacts

- [run_alignment_audit.py](run_alignment_audit.py) - the exact audit script used,
  unmodified
- [audit_summary.json](audit_summary.json) - top-level result: input shapes and
  scales, threshold sweep statistics, orientation winners per metric, mean IoU by
  orientation, representative-threshold selection rule, candidate groupings,
  recorded conclusion status, and limitations
- [orientation_comparison.json](orientation_comparison.json) - the full 28-row
  per-threshold, per-orientation comparison with IoU, precision, recall, median and
  p90 contour distances in both directions, and boundary pixel counts and strides
- [candidate_metrics.json](candidate_metrics.json) - per-candidate record for the
  eight inspected candidates, nested by threshold, with the bbox-proxy note
- [candidate_metrics.csv](candidate_metrics.csv) - the same candidate data
  flattened to 56 rows, one per candidate and threshold
- [mask_alignment_overview.png](mask_alignment_overview.png) - the principal
  figure: projected `tifxyz`-invalid contour against the render edge-connected
  background contour over the processed raster
- [candidate_audit_crops.png](candidate_audit_crops.png) - per-candidate crops for
  the eight inspected candidates, labelled with rank, component ID, group, score,
  distances and overlap

Not stored in this repository:

- the source render JPG
- the normalized `tifxyz` TIFF arrays

The committed outputs preserve the completed comparison. Rerunning the audit
requires both external inputs at the paths recorded in `audit_summary.json`, so
this is not a full reproduction from a Git clone alone

## Limitations

- one render
- one normalized `tifxyz` parameterization
- four simple orientation hypotheses only
- no transpose and no arbitrary registration search
- render-background extraction depends on the grayscale threshold and the
  minimum component-size setting, which are explicit run parameters and not
  universally calibrated values
- candidate footprints are bounding-box approximations, not component pixel masks
- the projected `tifxyz` mask failed the audit's own strict reconstruction check
- mask agreement does not establish CT array indexing
- the audit is not CT validation. It compares two 2D masks and carries no CT
  evidence and no surface-normal geometry

## Attribution

Source render derived from Vesuvius Challenge open data (PHercParis4 segment
20260623163339-w110-112). This audit does not imply endorsement by the Vesuvius
Challenge

- Author: Olga Ivanova
- Repository: https://github.com/olgaiv39/scroll-anchor
