# PHerc render ranking-policy audit

Offline comparison of candidate **review order** for the PHercParis4 w110-112
exploratory render run.

## Purpose

Render-background diagnostics showed that several top-ranked `analyze-render`
candidates sit largely on black render background rather than on papyrus texture.
This audit asked a single question: would a simple background-aware secondary
score put a better set of candidates in front of a human reviewer first?

The audit reused the already exported combined-diagnostics run. It read only
`regions.json`, `metadata.json` and `summary.json`, plus the source render once for
the contact-sheet crops.

The audit did **not**:

- rerun candidate detection
- change the detector score
- change `analyze-render` outputs
- classify any candidate as a confirmed error or a confirmed false positive
- introduce production ranking behavior

Four quantities are kept strictly separate throughout:

| Quantity | Meaning | Status |
| --- | --- | --- |
| detector score | `score` produced by `analyze-render` | authoritative, unchanged |
| diagnostic overlap | `render_background_overlap_fraction` | diagnostic only, no effect on scoring |
| secondary review score | the penalty formulas below | hypothetical, not implemented |
| exported candidate order | the ranked subset in `regions.json` | authoritative, unchanged |

## Policies compared

With `overlap = render_background_overlap_fraction`, clamped defensively to
`[0, 1]` (no stored value was outside that range):

- **raw ranking**
  `raw_score`
- **half penalty**
  `raw_score * (1 - 0.5 * render_background_overlap_fraction)`
- **full penalty**
  `raw_score * (1 - render_background_overlap_fraction)`
- **hard 0.20 split**
  candidates with `overlap < 0.20` first in raw-score order, then the rest in
  raw-score order. Comparison only, not proposed for production.

Ties are broken deterministically by higher adjusted score, then higher raw score,
then lower component ID.

## Main findings

Input distribution across the 200 exported candidates:

- 200 exported candidates were compared
- 133 candidates had exactly zero render-background overlap
- 58 had overlap at least 0.20
- 39 had overlap at least 0.50

Effect on the top of the review list:

- the raw top 20 contained six known render-background calibration cases
- both the half and the full penalty removed those six cases from the top 20
- the half and full penalties produced the same top-20 and top-50 candidate sets,
  so the stronger penalty changed nothing at the depth a reviewer actually works
- the half penalty was therefore the least aggressive sufficient tested soft policy
- interior controls at original ranks 7 and 11 remained prominent, moving to
  ranks 5 and 7

Scale of global reordering under the half penalty:

- Spearman rank correlation with raw order: 0.599
- median absolute rank movement: 26
- 129 of 200 candidates moved by more than 20 positions

That last group of numbers is the reason for caution. The reordering is large
because the overlap distribution is strongly bimodal: most candidates carry no
penalty at all, and the penalized ones fall a long way, displacing everything
below them upward. Zero-overlap candidates keep their raw score, so their order
relative to one another is preserved and none of them is demoted. Even so, a
policy that moves two thirds of the list by more than 20 positions is not
something to adopt on the strength of one render.

## Decision

**Status: evaluation completed; production policy not adopted**

- the audit itself is complete
- render-background overlap is a useful reviewer signal
- no automatic penalty or re-ranking was added to ScrollAnchor
- the raw detector score and the exported order remain authoritative
- the half-penalty formula remains an experimental secondary review-order
  hypothesis, not a detector change and not a replacement for the detector score
- more renders and independent expert-labelled candidates are required before
  production adoption

The practical takeaway for a reviewer today is to read
`render_background_overlap_fraction` alongside the score in `regions.json` and use
it to decide where to spend attention. That requires no change to the tool.

## Limitations

- one render
- six background calibration examples
- two interior controls
- one ambiguous example
- no independent ground-truth set; the calibration cases were selected by
  inspecting this same run and are not an independent validation set
- overlap does not fully model adjacency: a candidate can sit immediately beside
  the background, or touch it, while recording little or no overlap, and such a
  candidate receives little or no penalty
- run-specific render-background threshold and component-size parameters
  (8-bit grayscale threshold 32, minimum component 5000 processed pixels,
  4-connectivity, edge-connected components only), which are not universally
  calibrated values
- the audit covers only the 200 exported candidates, not the 428 above-threshold
  components or the 1152 total components
- the penalty coefficients were fixed in advance; no parameter search was
  performed, so 0.5 is not claimed to be optimal
- candidates remain exploratory 2D visual anomalies for manual review

Visible black render background and `tifxyz`-invalid surface are different things
that can disagree over large areas. This audit used the render-background
diagnostic only; `tifxyz` validity diagnostics were deliberately not used.

## Artifacts

- [run_ranking_policy_audit.py](run_ranking_policy_audit.py) - the exact analysis
  script used for this audit
- [audit_summary.json](audit_summary.json) - verified input counts, policy
  formulas, top-k statistics, calibration rank table, rank-movement statistics,
  decision criteria, recommendation, rationale, and limitations
- [ranking_comparison.csv](ranking_comparison.csv) - one row per candidate, with
  component ID, raw rank and score, overlap, background distance, touch flag, the
  half and full scores and ranks, the hard-split rank, and the calibration group
- [ranking_comparison.json](ranking_comparison.json) - the same comparison with
  per-candidate bounding boxes, centroids, and per-policy rank changes
- [top20_raw.png](top20_raw.png) - contact sheet of the top 20 under raw ranking
- [top20_half_penalty.png](top20_half_penalty.png) - top 20 under the half penalty
- [top20_full_penalty.png](top20_full_penalty.png) - top 20 under the full penalty

In the contact sheets, the yellow box is the candidate bounding box and the frame
color marks the calibration group. Each crop is labelled with policy rank,
original rank, component ID, raw score, secondary review score, overlap,
background distance, and the source-JPG bounding box.

Notes on reproducing this:

- the JSON inputs this audit consumed are now published. The combined-diagnostics
  run supplying `regions.json`, `metadata.json` and `summary.json`, including the
  `render_background_overlap_fraction` field the script reads, is committed at
  [`../pherc-render-combined-diagnostics/`](../pherc-render-combined-diagnostics/README.md).
  Note that the older [`results/pherc-render/`](../pherc-render/README.md) run
  predates the render-background diagnostics and does not carry those fields
- the source JPG remains external and is not stored in this repository. It is still
  required to regenerate the contact-sheet crops, so the analysis numbers can be
  recomputed from the published JSON while the image outputs cannot. This is not a
  full reproduction from a Git clone alone
- the script is the exact analysis script used for this audit, unmodified. It
  expects the inputs at the paths recorded in `audit_summary.json`
- the committed JSON, CSV, and contact sheets preserve the completed result even
  when those large source inputs are unavailable

## Attribution

Source render derived from Vesuvius Challenge open data (PHercParis4 segment
20260623163339-w110-112). This review does not imply endorsement by the Vesuvius
Challenge.

- Author: Olga Ivanova
- Repository: https://github.com/olgaiv39/scroll-anchor
