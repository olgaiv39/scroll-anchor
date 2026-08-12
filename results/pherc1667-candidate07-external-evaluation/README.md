# PHerc1667 candidate-07: external evaluation of the frozen ScrollAnchor baseline

Read-only evaluation of the frozen default ScrollAnchor baseline against an independently labelled trace-divergence sector supplied by Alan Thompson

No detector code, threshold, configuration default, mask or normal-estimation rule was changed, tuned or added for this evaluation. The run below uses `configs/default.yaml` unmodified and `correction.enabled = false`

## Evaluation package

Alan Thompson (`altommo`) prepared the candidate-07 package used for this evaluation

It covers PHerc1667 cube `z09728_y05632_x10496`, using a 256³ crop of the 2.399 µm masked CT volume. The reference-divergence labels come from an independent VC3D candidate-tracing diagnosis rather than manual visual annotation

Package identity:

- `scrollanchor-package-pherc1667-candidate07.zip`
- received 2026-08-03
- 14,732,072 bytes
- SHA-256 `2460120eba5be9c0dde08827ffe8c5f196f14e1eb6db0fe60645751bd56871a6`

The filename, size and checksum were verified before the evaluation

Raw package files are not included in this repository. The CT ROI, tifxyz arrays, evaluation-label arrays and reproduced `.npy` outputs remain under the gitignored `data/external/pherc1667-candidate07/`

Only the derived evaluation artifacts are published here

## Reproduction commands

From the repository root, with the package extracted to
`data/external/pherc1667-candidate07/`:

```bash
# 1. Frozen baseline, unchanged defaults, no correction
scroll-anchor analyze \
  --surface data/external/pherc1667-candidate07/surface \
  --volume data/external/pherc1667-candidate07/ct_roi.npy \
  --config configs/default.yaml \
  --output data/external/pherc1667-candidate07/reproduced-default \
  --no-channels

# 2. Evaluation (read-only; recomputes every number in this directory)
python scripts/evaluate_pherc1667_candidate07.py
```

Repository state: branch `main`, HEAD `ae61e391736cbb64a38271a345fec0b8449e785a` ("add validated container workflow"), Python 3.8.15, numpy 1.23.5

## Reproduction vs Alan's first pass

The reproduced run matches Alan's first pass exactly. The resolved configuration, `diagnostics.json` summary, `review_regions.json` and all seven supplied per-vertex arrays are byte-identical, including NaN patterns; the maximum absolute difference is 0 for every finite entry

## Results

Positive class throughout: the supplied labelled divergence (`distance_to_reference > 4` against the accepted VC3D reference). This is an evaluation label, not physical-sheet ground truth - see Limitations. TP/FP/FN/TN below are standard confusion-matrix terms defined against this positive class

### Coverage (Domain A - all 100 surface-valid vertices)

| Quantity | Count | Fraction |
|---|---:|---:|
| Surface-valid vertices (tifxyz `z > 0`, rows 2-11 × cols 2-11) | 100 | 1.000 |
| Detector-evaluable (`surface.valid & normal_valid`) | 64 | 0.640 |
| Not evaluated (no detector decision emitted) | 36 | 0.360 |
| Reviewed | 64 | 0.640 |
| Evaluated but not reviewed | 0 | 0.000 |

| Group | Total | Reviewed | Evaluated, not reviewed | Not evaluated |
|---|---:|---:|---:|---:|
| Reference-divergence sector (`distance_to_reference > 4`) | 34 | 18 | 0 | 16 |
| Clean surface | 66 | 46 | 0 | 20 |

The 36 not-evaluated vertices are neither true negatives nor false negatives: the detector emitted no decision for them, so a zero in the review mask there reflects absence of coverage, not a negative call

### Domain B - detector-evaluable (n = 64), the only domain expressing detector decisions

| TP | FP | FN | TN | Precision | Recall | F1 | IoU | Dice |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 46 | 0 | 0 | 0.281 | 1.000 | 0.439 | 0.281 | 0.439 |

- Clean false positives: 46 of 46, for a clean false-positive rate of 1.000
- Evaluable failure sector reviewed: 1.000
- Evaluable clean surface left unflagged: 0.000

Recall 1.000 and IoU 0.281 here are not measures of localisation. Every evaluable vertex was flagged, so precision, IoU and Dice are fully determined by labelled-divergence prevalence in the covered block (18/64 = 0.281): a detector that flags everything scores identically

### Domain C - literal emitted mask (n = 100), coverage-confounded

| TP | FP | FN | TN | Precision | Recall | F1 | IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 46 | 16 | 20 | 0.281 | 0.529 | 0.367 | 0.225 |

All 16 FN and all 20 TN are exactly the uncovered vertices. This table credits and penalises the detector for decisions it never made, and is included only for completeness, not as the primary result

### Review-rule decomposition

Review rule (`report.apply_review`):
`valid & ((switch_score >= 0.5) | (confidence < 0.5) | (drift_score >= 0.35 * spacing))`

| Condition | True on evaluable (of 64) |
|---|---:|
| `switch_score >= 0.5` | 64 |
| `confidence < 0.5` | 16 |
| `drift_score >= 2.8` (0.35 × spacing 8.0) | 42 |
| Reviewed vertices not satisfying switch condition (`review & ~switch_condition`) | 0 |
| Would remain reviewed if the switch rule were removed (`(confidence<0.5)\|(drift>=2.8)`) | 50 |

The switch condition alone reproduces the frozen review mask exactly: `switch_score` takes the single value 1.0 at all 64 evaluable vertices, so `review & ~switch_condition` is 0 by construction. That says the switch condition is sufficient to force full review, not that the other two conditions are weak on their own

To check that separately, we compute a diagnostic counterfactual: what the review OR-rule would flag with the switch term removed, i.e. `(confidence < 0.5) | (drift_score >= 0.35 * spacing)` evaluated on the 64 detector-evaluable vertices. This is not an alternative detector, only a decomposition of the frozen rule, computed from the existing per-vertex arrays with no threshold changed

| Reviewed | TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 / 64 | 15 | 35 | 3 | 11 | 0.300 | 0.833 | 0.441 |

The frozen final review mask stays 64/64 either way. Switch saturation alone is sufficient to reproduce it, but it is not the only source of over-review: confidence and drift alone would still flag 50 of 64 evaluable vertices, including 35 of 46 clean evaluable vertices (11 of 46 clean evaluable vertices are left unflagged)

The saturated `switch_score` comes from `diagnostics._robust_residual_magnitude`: the reference surface is a `median_filter` of size `switch_smooth_window = 31` over a 14×14 grid. The window exceeds the grid, so with `mode="nearest"` the "local" reference collapses to a single constant point. The sentinel-filled invalid vertices (value `-1`, 96 of 196 = 49% of the grid) are included in that median and replicated by the border mode, so the reference lands exactly on `(-1, -1, -1)`, outside the patch. `switch_mag` then measures distance from the patch to that point, giving `switch_ratio` between 32.8 and 43.7 against a threshold of 0.5

For contrast, `_grid_normal_residual` (which produces `geom_offset`) masks invalid vertices before smoothing; `_robust_residual_magnitude` applies `valid` only after computing the magnitude

### The 100 vs 64 coverage gap

`pipeline.analyze_surface` evaluates `surface.valid & normal_valid`. In `normals.compute_normals`, invalid vertices are set to `NaN` before `np.gradient`. The central- and one-sided-difference stencils propagate that `NaN` to every vertex whose 3×3 neighbourhood touches an invalid vertex, so the valid block is eroded by exactly one vertex on every side

Verified three independent ways: the evaluable mask equals a 3×3 binary erosion of the surface-valid block, it equals rows 3-10 × cols 3-10 (inner 8×8 = 64), and it equals the set of vertices with finite emitted `chosen_offset`

The excluded ring is exactly generation 5 (all 36 vertices), and all 16 uncovered labelled-divergence vertices are generation 5:

| Generation | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Surface-valid | 4 | 12 | 20 | 28 | 36 |
| Detector-evaluable | 4 | 12 | 20 | 28 | 0 |
| Labelled-divergence | 0 | 1 | 5 | 12 | 16 |
| Labelled-divergence and uncovered | 0 | 0 | 0 | 0 | 16 |

This confirms the preliminary inspection independently: the outer surface-valid ring is excluded by normal validity, and 16 of the 34 labelled-divergence vertices fall in it. With this validity topology and the current normal computation, the entire outermost traced generation is not detector-evaluable, and on this package that is where 47% of the labelled divergence lies

### Signal comparison (detector-evaluable vertices only, n = 64)

| Comparison | Pearson r | Spearman ρ | MAE (vox) |
|---|---:|---:|---:|
| `geom_offset` vs `signed_normal_offset` | 0.043 | 0.091 | 2.31 |
| `abs(geom_offset)` vs `distance_to_reference` | 0.488 | 0.467 | 1.88 |

`geom_offset` is essentially uncorrelated with the supplied signed reference offset. Sign agreement is 0.578, close to chance, and is not evidence of physical agreement: the package states its signed-offset sign is comparable only within that one trace, and ScrollAnchor fixes normal orientation by its own global mean-normal convention

`abs(geom_offset)` shows a moderate positive rank association with `distance_to_reference`, but its magnitude is heavily compressed: median 0.50 vs 2.49 voxels, maximum 4.17 vs 6.31 voxels on the evaluable set. This matches the package's own note that the failure "looks switch-like": the trace stays near bright material, so a label-to-nearest-sheet offset reads small even where the trace is far from the reference

Per-channel distributions, labelled divergence vs clean (evaluable only):

| Channel | Labelled-divergence mean / median | Clean mean / median |
|---|---:|---:|
| `confidence` | 0.681 / 0.739 | 0.721 / 0.769 |
| `drift_score` | 4.833 / 4.750 | 4.109 / 3.500 |
| `switch_score` | 1.000 / 1.000 | 1.000 / 1.000 |
| `chosen_offset` | 1.417 / 3.750 | −2.717 / −3.000 |
| `geom_offset` | 0.493 / 0.733 | 0.093 / 0.027 |

None of these channel summaries establish reliable discrimination on this patch. `switch_score` is identical in both groups; `confidence` is slightly lower on labelled-divergence vertices (0.681 vs 0.721) and `drift_score` slightly higher (4.83 vs 4.11), both in the direction a working signal would need, but the differences are small relative to the spread and the distributions overlap heavily. That does not mean the signal is absent, only that these particular summaries do not discriminate on this patch

## Interpretation

The frozen baseline flagged all 64 evaluable vertices as one `sheet_switch` region and emitted no decision for the remaining 36. It reviewed 18 of 34 labelled-divergence vertices and all 46 evaluable clean vertices, leaving 16 labelled-divergence vertices uncovered

The result reflects coverage loss and saturated over-review rather than successful localisation. Two observed failure mechanisms happen to co-occur here: normal-validity erosion removes the entire outermost generation, including 47% of the labelled sector, and a degenerate switch reference fires at every remaining vertex. Switch saturation alone is enough to explain the all-flagged review mask, but removing it diagnostically still leaves confidence and drift flagging 50 of 64 vertices, so switch saturation is not the whole story

The baseline did not demonstrate localisation on this package. It also did not demonstrate a false-negative failure on the covered set, since it flagged everything. The emitted mask is therefore uninformative for localisation on this patch

The final binary review mask and `switch_score` carry no localisation information on this patch. Other continuous channels retain some variation, and the diagnostic removal of the switch rule produces some non-constant decisions, but this single case does not establish reliable per-vertex discrimination

## Limitations

- The reference-divergence label marks divergence of this trace from an accepted VC3D reference; it is not physical-sheet ground truth and does not prove which physical sheet is correct or that a permanent sheet switch occurred
- The package explicitly withdraws the `w011` accepted surface as an evaluation reference, so the reference behind `distance_to_reference` is not re-verifiable from the package contents alone
- n = 100 surface-valid vertices, one 14×14 patch, one cube, one scroll - all rates carry wide binomial uncertainty and nothing here generalises
- The trace is coarse (step size 10 voxels); a 34-vertex sector is near the resolution limit for a localiser, as the package itself notes
- Domain B precision/IoU/Dice measure labelled-divergence prevalence in the covered block, not discrimination
- The switch channel is saturated, so any correlation involving it is meaningless
- No confidence intervals or significance tests are reported; they would not add meaningful inferential evidence on a single fully flagged, saturated patch

### Limits of interpretation

- localisation of the full reference-divergence sector
- that flagged vertices are physically incorrect sheet assignments
- behaviour of the 16 vertices outside detector coverage
- equivalence of `geom_offset` and `signed_normal_offset`
- generalisation to other scrolls, cubes, traces or grid sizes
- physical correctness of the accepted reference

## Status and next validation step

Candidate-07 now serves as a development case rather than independent validation for a future detector change:

- the frozen baseline does not localise the labelled divergence
- it exposes both coverage loss and strong over-review on a small real patch
- any detector change motivated by this case needs to be evaluated separately
- final validation should use a second blind patch whose labels stay withheld until after ScrollAnchor's output is frozen

## Files

| File | Contents |
|---|---|
| `evaluation_summary.json` | Provenance, input validation, resolved frozen config, reproduction comparison, all three domains, signal statistics, limitations |
| `per_vertex_metrics.csv` | One row per surface-valid vertex (100). Detector columns are blank where no decision was emitted, never filled with 0 |
| `overview.png` | Reference-divergence sector, detector-evaluable mask, review mask, outcome map |
