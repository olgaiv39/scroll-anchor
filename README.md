# ScrollAnchor

**Conservative, read-only surface-label diagnostics for volumetric papyrus CT**

ScrollAnchor takes an *approximate* papyrus surface (Volume Cartographer
`tifxyz`) and a CT volume/ROI, and produces diagnostics for two high-value failure
modes:

1. **Normal-direction drift** - the surface sits off the physical sheet
2. **Sheet-switch jumps** - a patch has jumped onto a neighboring sheet (the most
   harmful error: it *looks* fine because it sits on a real sheet)

It emits per-vertex confidence, confidence-ranked **review regions**, and
machine-readable reports. It is **diagnostics-first**: it flags, it does not move
labels. Conservative correction proposals are available but **off by default**

ScrollAnchor is an early open-source research tool developed for the Vesuvius
Challenge 2026 open problems around surface-label quality, mesh-tracing errors, and
neighboring-sheet switches (see `scrollprize.org/2026_open_problems`, problems on
*Surface Prediction & Topology*, *Mesh Tracing Failures*, and *Label Quality &
Imprecision*). It is functional, reproducible research software with a clear
validation roadmap, useful today for expert-assisted surface review, controlled
benchmark construction, and identifying potentially high-risk surface regions

## Choose a workflow

The core workflow is the 3D surface-label and CT diagnostic workflow. The 2D render
workflow is exploratory review prioritization and is not volumetric validation. The
PHerc artifacts are a real-data case study and evidence chain rather than a separate
product, and the render candidates in them are not confirmed reconstruction failures

| Available data | Command | Purpose |
| --- | --- | --- |
| Surface labels or tifxyz plus a CT volume or ROI | `scroll-anchor analyze` | Run the core 3D drift and neighboring-sheet diagnostics |
| A 2D render image without CT evidence | `scroll-anchor analyze-render` | Export exploratory review candidates |
| Existing render-analysis outputs | `scroll-anchor render-report` | Build a manual review report without rerunning detection |
| Render candidates plus tifxyz channels | `scroll-anchor map-render-candidates` | Map selected candidate ranks to surface coordinates |

Mapping selects candidates by exported rank, which is not the component ID

## Why this, and how it relates to the existing ecosystem

Verified against `ScrollPrize/villa`:

- `lasagna` **corrects/grows** surfaces via GPU-oriented optimization (needs
  preprocessed evidence + winding volumes). ScrollAnchor is a **read-only
  pre-filter** that says *where* correction is safe versus risky
- `segmentation/vc_proofreader` is a **human** napari review UI with no automatic
  error localization. ScrollAnchor prioritizes *which patches a human should open*
- `segmentation/evaluation` computes **global** metrics against full ground truth.
  ScrollAnchor is **localized** and needs **no ground truth** at inference

The gap it fills: a standalone, CPU-friendly, ground-truth-free tool that turns
`(tifxyz surface + CT ROI)` into localized drift/switch diagnostics

## Validation status

Validation is layered, and each layer supports a different claim:

- the synthetic benchmark demonstrates precise detection under **controlled
  synthetic** corruptions of a gently curved multi-sheet volume
- the real-cube experiment uses real CT and real sheet geometry with injected
  corruptions and supports the conservative safety concept
- naturally occurring real annotation failures have **not yet been evaluated**
  against full ground truth

The exploratory 2D PHerc workflow provides review candidates and supporting
evidence, not volumetric detector validation

## Experimental relative synchronization

`scroll_anchor.synchronization` provides a generic CPU CP-SAT surface for synchronizing compatible local sheet-correspondence hypotheses into coherent relative phase structure. It is ground-truth-free at inference and retains the exact relative-gauge quotient and exact endpoint/component-use conflict encoding

`section_local_phase_patch_score` is a threshold-free descriptive score for selected non-modal phase structure. In a 20-case external evaluation prepared by altommo, relative synchronization reached AUROC 0.672 and Average Precision 0.385. This is evidence of meaningful correspondence structure, not a finished sheet-switch classifier, and the Alan evaluation is not a new blind validation

See the [short external result note](results/pherc-alan-relative-synchronization-20260831/SHORT_REPORT.md)

## Results (synthetic benchmark, reproducible)

Two parallel-sheet CT volume + a clean surface corrupted with drift, a
sheet-switch, an ambiguous low-contrast+drift zone, and a hole. Mean over 5 seeds
(`80x80` grid), default config:

| Metric | Value |
|---|---|
| Sheet-switch detection precision / recall | **1.00 / 1.00** |
| Drift displacement recovery MAE | **~0.56 voxels** |
| **Harmful acceptance - trust labels as-is** | 0.051 |
| **Harmful acceptance - naive snap-to-brightest** | ~0.60 |
| **Harmful acceptance - ScrollAnchor** | **0.00** |
| Clean-region stability (not needlessly flagged) | **1.00** |
| Review fraction | ~0.12 |

**Harmful acceptance** = fraction of vertices a method confidently accepts (keeps
or moves) that end up on the *wrong sheet*. This is the primary metric and
encodes the core safety principle: when evidence is ambiguous, flag for review -
never emit a confident label on the wrong sheet

Reproduce:

```bash
pip install -e .
scroll-anchor benchmark --output results/bench --seed 0
cat results/bench/metrics.json
```

## Real-cube benchmark (real CT + real geometry, controlled corruptions)

`scripts/run_real_cube_benchmark.py` runs the diagnostics on a real Scroll 1
instance-label cube (`02256_02512_04816`): a medial surface is extracted from one
labelled sheet, then **controlled** drift and a **real neighbouring-sheet** switch
are injected. This is *not* validation on naturally occurring annotation errors

```bash
pip install -e ".[benchmark]"
python scripts/run_real_cube_benchmark.py --output results/real_cube_02256_02512_04816
```

Findings on this cube (source sheet 328, neighbour 329, 96³ ROI):

- **Conservative safety behaviour transferred to the tested cube.** ScrollAnchor's
  harmful acceptance (confidently accepting a vertex that sits on the wrong sheet) is
  **0.00** versus **~0.15** for naive snap-to-brightest; switch review-recall is
  **1.00** - the injected neighboring-sheet switch is always surfaced for review, and
  no wrong-sheet vertex is confidently accepted
- **Precision is currently limited on the tested strongly curved real geometry.**
  Thresholds calibrated on flat synthetic sheets over-fire on real papyrus curvature:
  switch precision ~0.19, drift localization remains weak (F1 ~0.01), and ~27% of the
  *clean* surface is flagged for review. On strongly curved real geometry the tool
  currently behaves as a very conservative "flag-for-review" filter rather than a
  precise localizer

On the tested cube this experiment **supports the conservative safety concept on
this controlled real-geometry benchmark** and illustrates a viable
expert-in-the-loop workflow - the injected switch is always surfaced and nothing
wrong-sheet is confidently accepted. It also usefully identifies the next research
bottleneck: real curvature increases false positives, so **curvature-aware residual
modelling and improved calibration** are the next development priorities. This
single controlled-corruption cube does not, on its own, establish general
real-scroll precision

## Install

```bash
pip install -e .            # CPU-only: numpy, scipy, tifffile, pyyaml
pip install -e ".[remote]"  # + zarr/fsspec to stream real CT ROIs over HTTP/S3
```

To run the full test suite, install the dev and benchmark extras - the NRRD-related
tests require the benchmark dependencies (pynrrd) and are skipped otherwise:

```bash
pip install -e ".[dev,benchmark]"
python -m pytest -q
```

### Container

The repository includes a CPU-only Dockerfile for reproducible runs. Build the
image from the repository root:

```bash
docker build -t scroll-anchor:local .
```

The entrypoint is the `scroll-anchor` CLI, so running the image with no arguments
prints the help screen:

```bash
docker run --rm scroll-anchor:local
```

Output directories require a writable mount. The synthetic benchmark writes into
the mounted path:

```bash
mkdir -p results/docker-bench

docker run --rm \
  -v "$PWD/results/docker-bench:/output" \
  scroll-anchor:local \
  benchmark --output /output --seed 0
```

Commands that use local files follow the same mount pattern. Mount input data
read-only where possible, and mount a separate writable directory for output:

```bash
docker run --rm \
  -v "/path/to/input:/data:ro" \
  -v "/path/to/output:/output" \
  scroll-anchor:local \
  <command> <arguments>
```

Manually validated in Docker: the image build, the CLI help screen, the runtime
extras, all subcommand help screens, and one synthetic benchmark run

Not validated in Docker: real CT analysis, remote Zarr access, real render
analysis, PDF generation, candidate mapping on real artifacts, and the full
real-cube benchmark

## Analyze a real surface

```bash
scroll-anchor analyze \
  --surface path/to/segment_tifxyz/ \
  --volume  path/to/volume.zarr \
  --config  configs/default.yaml \
  --output  results/run/
```

- `--volume` accepts a local `.npy` (in-memory, for tests) or a zarr path/URL
  indexed `[z, y, x]`. For zarr, only the surface's bounding box (+margin) is
  read into memory, so this is ROI-scoped and memory-safe
- Add `--enable-correction` to also emit conservative, gated correction proposals

Outputs:

```
results/run/
├── diagnostics.json        # summary stats + resolved config
├── review_regions.json     # prioritised, clustered regions to inspect
├── arrays/*.npy            # confidence, selection provenance, review causes, ...
└── surface/                # tifxyz copy + diagnostic/review-cause channels
```

## How it works (brief)

For each surface vertex: estimate the world-space normal, sample the CT intensity
profile along ±`radius` voxels (trilinear, CPU), then:

- **Drift** = signed offset to the distance-weighted nearest sheet peak
- **Sheet-switch** = a robust (median-consensus) ~one-spacing positional jump,
  confirmed by strong on-sheet evidence, grown by hysteresis over the patch
- **Confidence** = product of local profile contrast, peak margin, and peak
  evidence - so any CT-profile weakness drives confidence toward 0. Geometric
  residual remains exported as a separate surface-geometry diagnostic.
- **Profile selection state** records whether the reference came from one or
  multiple detected local peaks, a forced global-maximum fallback, an unusable
  profile, or a vertex outside detector coverage. It describes provenance and
  does not by itself trigger review.
- **Review** = switch or low confidence. Drift remains exported as an exploratory
  diagnostic, but is not a default review trigger because its real-geometry
  sheet-specificity has not been established. Set
  `review.include_drift_in_review: true` only to reproduce the legacy policy.
  The `review` array remains their combined actionable mask; additive
  `review_low_confidence` and `review_switch` boolean arrays identify each
  cause (and may both be true). With legacy drift review enabled,
  `review_drift` records its additional cause.

See `docs/method.md` for details and `docs/coordinate_conventions.md` for the
coordinate/normal conventions (verified against `villa/lasagna/tifxyz_format.md`
and the `vesuvius` tifxyz API)

## Exploratory 2D render analysis (separate workflow)

The `analyze` command above is the primary 3D pipeline: it needs a `tifxyz` surface
and a CT volume/ROI, and reports drift and sheet-switch diagnostics with
through-thickness CT evidence and surface-normal geometry

`analyze-render` is a **separate, exploratory** workflow for the case where all you
have is a single downsampled 2D surface **render** (a JPG), with no surface geometry
and no CT volume. It runs a CPU-only, classical image-processing detector that flags
**candidate visual discontinuities** - places where the render texture appears to
shift laterally, which *may* correspond to a sheet skip or a local render shift

```bash
pip install -e ".[render]"   # + Pillow for JPG decode and overlay drawing
scroll-anchor analyze-render \
  --render path/to/surface_render_ds8.jpg \
  --output results/render-run/ \
  --working-downsample 2
```

- Memory policy: the JPG is never decoded at full resolution. Pillow `draft()`
  downscales at the decoder, then `--working-downsample` reduces it further; a
  `--max-pixels` budget aborts before an unsafe allocation
- Coordinates: every region carries JPG row/col plus a *mapped* full-render row/col
  (JPG coordinates multiplied by the documented `--full-render-factor`, default 8).
  The mapped coordinates are not verified VC3D coordinates

Outputs:

```
results/render-run/
├── overlay.png         # ranked, ID-labelled boxes over a readable copy of the render
├── regions.json        # candidate regions with scores, directions, coordinates
├── diagnostics.npz     # compact processed-resolution seam/anomaly/texture arrays
├── metadata.json       # shapes, scales, params, runtime, peak RSS, limitations
├── summary.json        # count funnel and exported score range (ranked subset)
├── top_candidates.png  # contact sheet of the top-ranked candidate crops
└── report.pdf          # multi-page review packet for manual inspection
```

### Regenerate the review PDF from an existing result directory

`render-report` rebuilds `report.pdf` (and `top_candidates.png` when the source
render is available) from artifacts that already exist. It is report-only: it reads
`metadata.json`, `summary.json`, `regions.json` and `overlay.png`, does **not** run
the detector, does **not** read `diagnostics.npz`, and overwrites only the report
files. Numerical results are unchanged

```bash
scroll-anchor render-report \
  --results results/render-run/ \
  --render path/to/surface_render_ds8.jpg   # optional, for larger crop pages
```

- With `--render`, crop pages are decoded fresh from the source JPG at the recorded
  working resolution and `top_candidates.png` is regenerated
- Without `--render`, the existing `top_candidates.png` is reused and the PDF notes
  that the source render was unavailable

## PHerc case study: one real render

This is a real-data case study built from one PHerc render. It links exploratory 2D
candidates to boundary diagnostics, orientation testing, tifxyz mapping, and limited
local CT inspection. It does not classify any candidate as a confirmed reconstruction
failure, and it does not constitute full volumetric validation

A published example run (PHercParis4 segment w110-112) is recorded in
[`results/pherc-render/`](results/pherc-render/README.md)

### PHerc evidence chain

The PHerc artifacts are a real-data case study on a single render, not a separate
product. They form one chain:

- render analysis exports exploratory 2D candidates
- [combined diagnostics](results/pherc-render-combined-diagnostics/README.md)
  describe tifxyz-invalid and render-background relationships without changing
  detector scores
- the [alignment audit](results/pherc-alignment-audit/README.md) compares four simple
  orientation hypotheses, with identity strongest among those tested but not proven
- the [ranking-policy audit](results/pherc-ranking-policy-audit/README.md) tests
  alternative review-order policies without changing the production ranking
- [candidate mapping](results/pherc-candidate-tifxyz-mapping/README.md) converts
  selected exported ranks to tifxyz surface coordinates under an explicit
  identity-orientation assumption
- [local CT evidence](results/pherc-ct-local-evidence/README.md) inspects ranks 7 and
  11 under a registration-supported transformed-coordinate interpretation that remains
  supported, not globally verified

No step in this chain classifies any candidate as a confirmed sheet skip, a
reconstruction error, or a false positive

### Published combined-diagnostics run

A lightweight combined run on the same PHerc render is now published, carrying both
diagnostic systems at once: normalized `tifxyz` valid-surface diagnostics and
render-derived edge-connected near-black background diagnostics. Both are reviewer
fields only, so raw scores and the exported candidate order are unchanged

Only the machine-readable core is committed (`regions.json`, `metadata.json`,
`summary.json`, plus the contact sheet); `diagnostics.npz`, the full overlay, and the
source JPG are not. Those candidates are the shared input behind the ranking-policy
and candidate-to-tifxyz artifacts below:
[`results/pherc-render-combined-diagnostics/`](results/pherc-render-combined-diagnostics/README.md)

### Render-to-tifxyz alignment audit

The two diagnostic masks above disagreed, so a separate offline audit compared
identity against horizontal, vertical, and combined flip hypotheses to check whether
that gap was an orientation error. Identity had the strongest measured agreement on
every metric (mean IoU `0.4348`, against `0.3435`, `0.1995` and `0.2008`)

The masks are not equivalent: `tifxyz`-invalid surface was a **subset** of a broader
render-background mask covering more than twice the area (`0.1338` against `0.3096`
at threshold 32), with recall `1.0`. That supports different validity semantics
rather than a misalignment. This makes identity the best tested hypothesis but does
not remove the explicit assumption, so `map-render-candidates` still requires
`--assume-identity-orientation`:
[`results/pherc-alignment-audit/`](results/pherc-alignment-audit/README.md)

### Render-background ranking-policy audit

Render-background diagnostics exposed several top-ranked candidates dominated by
black render background rather than papyrus texture. An offline audit compared
raw, half-penalty, full-penalty, and hard-split review orders on an already
exported run, without rerunning detection. The half penalty
(`raw_score * (1 - 0.5 * overlap)`) was the least aggressive sufficient tested
policy: it cleared the known background cases out of the top 20 while keeping the
interior controls prominent

It was **not adopted**, because the evaluation used a single render and
substantially reordered the full candidate list (Spearman 0.599 against raw
order; 129 of 200 candidates moved by more than 20 positions). Runtime scores and
ranking are unchanged; render-background overlap stays a diagnostic field for
reviewers to read alongside the score. Details, artifacts, and limitations:
[`results/pherc-ranking-policy-audit/`](results/pherc-ranking-policy-audit/README.md)

### Candidate-to-tifxyz mapping

`map-render-candidates` is the reproducible bridge from an exported 2D candidate to
normalized `tifxyz` surface coordinates. It selects candidates by exported rank or
component ID (never confusing the two), maps the source-render centroid onto the
tifxyz grid by aligned half-open containment, and reads `x.tif`, `y.tif`, `z.tif` at
that cell

```bash
scroll-anchor map-render-candidates \
  --regions results/render-run/regions.json \
  --metadata results/render-run/metadata.json \
  --tifxyz-dir path/to/tifxyz_normalized \
  --rank 7 --rank 11 \
  --assume-identity-orientation \
  --output results/mapping-run/
```

- strict shape validation: scale factors are derived from the input shapes and must
  be exact positive integers per axis; incompatible rasters fail
- dual agreement: the cell is derived independently from the source and processed
  centroids and the two must match
- identity render-to-tifxyz orientation must be explicitly acknowledged with
  `--assume-identity-orientation`; it is never assumed silently and the command
  cannot verify it
- it does **not** assume CT array order and performs no CT validation. The values
  are raw tifxyz coordinates, not CT NumPy indices
- on the published PHerc run, ranks 7 and 11 mapped to valid tifxyz cells with both
  coordinate paths in agreement:
  [`results/pherc-candidate-tifxyz-mapping/`](results/pherc-candidate-tifxyz-mapping/README.md)

What this workflow **cannot** report, by construction: confirmed sheet switches,
true 3D drift, signed error along a surface normal, voxel displacement, or corrected
surface coordinates. A flat render has no through-thickness CT evidence and no
normal geometry, so its output is a set of 2D candidates for manual community
review, not a validated correction. If nothing passes the conservative texture,
border, multi-scale, and coherence gates, it returns an empty region list rather
than forcing a positive

### Local CT evidence for the mapped candidates

Small level-0 CT neighbourhoods were extracted at the positions that exported ranks 7
and 11 map to, so the mapped candidates can be looked at against real volumetric data.
The tifxyz coordinates were carried into the CT volume frame by the inverse
registration; that transformed interpretation is supported - not verified - by the
registration landmark residuals and by a native-versus-transformed chunk-presence
check. Each crop is 64x64x64 uint8, centred on the mapped voxel

Both mapped points fall within populated regions of the masked reconstruction, where
laminar structure is visible, and the two ranks sit in measurably different local
intensity contexts (rank 7 in a lower-density neighbourhood than its crop, rank 11 in a
higher-density one). This is evidence for inspection only: it does not classify either
candidate and it does not validate the detector. The coordinate-frame interpretation and
the identity render-to-tifxyz orientation assumption both remain open

- [`results/pherc-ct-local-evidence/`](results/pherc-ct-local-evidence/README.md)
- [`rank-7-orthogonal.png`](results/pherc-ct-local-evidence/rank-7-orthogonal.png)
- [`rank-11-orthogonal.png`](results/pherc-ct-local-evidence/rank-11-orthogonal.png)

## Current scope and development priorities

These are the current boundaries of what has been demonstrated, and the research
priorities that follow from them:

- **Precise detection is established only for the synthetic benchmark.** Those
  results come from **controlled synthetic** corruptions of a gently curved
  multi-sheet volume, not from real annotation failures
- **The real-cube experiment uses real CT and real sheet geometry with injected
  corruptions.** It validates the conservative safety concept; it is *not* validation
  on naturally occurring annotation errors, which have **not yet been evaluated**
- **Real curvature currently increases false positives.** Thresholds tuned on flat
  synthetic sheets over-fire on strongly curved real papyrus (~27% of the clean
  surface flagged on the tested cube); curvature-aware residuals are a priority
- **Drift localization requires improvement** on real geometry (F1 ~0.01 on the
  tested cube), where genuinely ambiguous zones are surfaced through review rather
  than corrected
- **Switch detection should become less dependent on a predefined smoothing window.**
  `switch_smooth_window` **must exceed** the switched-patch diameter; too small a
  window silently lowers switch recall
- **Confidence calibration requires validation on additional cubes.** Normal
  estimation degrades at surface discontinuities; switch detection uses a 3D
  positional residual (not the normal projection) to stay robust to this
- **Direct VC3D coordinate alignment remains to be verified.** The real-cube surface
  is exported in a **cube-index coordinate frame** (ROI-local indices offset by the
  cube origin) with NRRD metadata validation / axis resolution enforced; full VC3D
  coordinate compatibility is not claimed until visual alignment is checked

Conclusion: ScrollAnchor is ready for technical community review as an experimental
diagnostic and validation framework. The current real-cube benchmark supports its
conservative safety principle and demonstrates a viable expert-in-the-loop workflow,
while also identifying precision on strongly curved surfaces as the main development
priority. The current release is most useful for assisted review, controlled
benchmark construction, failure analysis, and collaborative method development.
Additional validation with known real annotation failures is needed before
recommending broader or unattended use

## Development roadmap

Research and engineering directions (priorities, not delivery commitments):

1. Curvature-detrended local residuals
2. Multi-scale neighboring-sheet-switch detection
3. Improved confidence calibration on real surfaces
4. Validation on known naturally occurring failure regions
5. Direct VC3D coordinate-alignment verification
6. Integration feedback from annotation and proof-reading workflows

## Community review and validation

ScrollAnchor is currently seeking technical review from participants familiar with
Vesuvius Challenge surface extraction, annotation, proof-reading, and volumetric
papyrus CT data

The most useful feedback would include:

- whether the targeted drift and neighboring-sheet-switch failure modes match real
  annotation or tracing problems;
- known small regions containing naturally occurring failures;
- review of the real-cube benchmark construction and coordinate assumptions;
- inspection of generated confidence, drift, switch, and review fields;
- advice on integration with existing VC3D, proof-reading, or segmentation workflows;
- comparison with existing tools that may already address part of the problem

The current repository should be treated as a working research contribution open to
validation and refinement, not as a community-endorsed solution

## License

MIT. Interoperates with the Volume Cartographer `tifxyz` format; see
`docs/coordinate_conventions.md` for attribution of format details

## Author

Olga Ivanova - ivolga.vak@gmail.com
