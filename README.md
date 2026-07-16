# ScrollAnchor

**Conservative, read-only surface-label diagnostics for volumetric papyrus CT.**

ScrollAnchor takes an *approximate* papyrus surface (Volume Cartographer
`tifxyz`) and a CT volume/ROI, and localizes two high-value failure modes:

1. **Normal-direction drift** — the surface sits off the physical sheet.
2. **Sheet-switch jumps** — a patch has jumped onto a neighboring sheet (the most
   harmful error: it *looks* fine because it sits on a real sheet).

It emits per-vertex confidence, confidence-ranked **review regions**, and
machine-readable reports. It is **diagnostics-first**: it flags, it does not move
labels. Conservative correction proposals are available but **off by default**.

ScrollAnchor is an early open-source research tool developed for the Vesuvius
Challenge 2026 open problems around surface-label quality, mesh-tracing errors, and
neighboring-sheet switches (see `scrollprize.org/2026_open_problems`, problems on
*Surface Prediction & Topology*, *Mesh Tracing Failures*, and *Label Quality &
Imprecision*). It is functional, reproducible research software with a clear
validation roadmap, useful today for expert-assisted surface review, controlled
benchmark construction, and identifying potentially high-risk surface regions.

## Why this, and how it relates to the existing ecosystem

Verified against `ScrollPrize/villa`:

- `lasagna` **corrects/grows** surfaces via GPU-oriented optimization (needs
  preprocessed evidence + winding volumes). ScrollAnchor is a **read-only
  pre-filter** that says *where* correction is safe vs. risky.
- `segmentation/vc_proofreader` is a **human** napari review UI with no automatic
  error localization. ScrollAnchor prioritizes *which patches a human should open*.
- `segmentation/evaluation` computes **global** metrics against full ground truth.
  ScrollAnchor is **localized** and needs **no ground truth** at inference.

The gap it fills: a standalone, CPU-friendly, ground-truth-free tool that turns
`(tifxyz surface + CT ROI)` into localized drift/switch diagnostics.

## Results (synthetic benchmark, reproducible)

Two parallel-sheet CT volume + a clean surface corrupted with drift, a
sheet-switch, an ambiguous low-contrast+drift zone, and a hole. Mean over 5 seeds
(`80x80` grid), default config:

| Metric | Value |
|---|---|
| Sheet-switch detection precision / recall | **1.00 / 1.00** |
| Drift displacement recovery MAE | **~0.56 voxels** |
| **Harmful acceptance — trust labels as-is** | 0.051 |
| **Harmful acceptance — naive snap-to-brightest** | ~0.60 |
| **Harmful acceptance — ScrollAnchor** | **0.00** |
| Clean-region stability (not needlessly flagged) | **1.00** |
| Review fraction | ~0.12 |

**Harmful acceptance** = fraction of vertices a method confidently accepts (keeps
or moves) that end up on the *wrong sheet*. This is the primary metric and
encodes the core safety principle: when evidence is ambiguous, flag for review —
never emit a confident label on the wrong sheet.

Reproduce:

```bash
pip install -e .
scroll-anchor benchmark --output results/bench --seed 0
cat results/bench/metrics.json
```

## Install

```bash
pip install -e .            # CPU-only: numpy, scipy, tifffile, pyyaml
pip install -e ".[remote]"  # + zarr/fsspec to stream real CT ROIs over HTTP/S3
```

To run the full test suite, install the dev and benchmark extras — the NRRD-related
tests require the benchmark dependencies (pynrrd) and are skipped otherwise:

```bash
pip install -e ".[dev,benchmark]"
python -m pytest -q
```

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
  read into memory, so this is ROI-scoped and memory-safe.
- Add `--enable-correction` to also emit conservative, gated correction proposals.

Outputs:

```
results/run/
├── diagnostics.json        # summary stats + resolved config
├── review_regions.json     # prioritised, clustered regions to inspect
├── arrays/*.npy            # per-vertex fields (confidence, drift, switch, ...)
└── surface/                # tifxyz copy + sa_confidence/sa_drift/sa_switch/sa_review channels
```

## How it works (brief)

For each surface vertex: estimate the world-space normal, sample the CT intensity
profile along ±`radius` voxels (trilinear, CPU), then:

- **Drift** = signed offset to the distance-weighted nearest sheet peak.
- **Sheet-switch** = a robust (median-consensus) ~one-spacing positional jump,
  confirmed by strong on-sheet evidence, grown by hysteresis over the patch.
- **Confidence** = product of contrast, peak margin, geometric continuity, and
  evidence — so any single weakness drives confidence toward 0.
- **Review** = switch, or low confidence, or large drift.

See `docs/method.md` for details and `docs/coordinate_conventions.md` for the
coordinate/normal conventions (verified against `villa/lasagna/tifxyz_format.md`
and the `vesuvius` tifxyz API).

## Real-cube benchmark (real CT + real geometry, controlled corruptions)

`scripts/run_real_cube_benchmark.py` runs the diagnostics on a real Scroll 1
instance-label cube (`02256_02512_04816`): a medial surface is extracted from one
labelled sheet, then **controlled** drift and a **real neighbouring-sheet** switch
are injected. This is *not* validation on naturally occurring annotation errors.

```bash
pip install -e ".[benchmark]"
python scripts/run_real_cube_benchmark.py --output results/real_cube_02256_02512_04816
```

Findings on this cube (source sheet 328, neighbour 329, 96³ ROI):

- **Conservative safety behaviour transferred to the tested cube.** ScrollAnchor's
  harmful acceptance (confidently accepting a vertex that sits on the wrong sheet) is
  **0.00** vs **~0.15** for naive snap-to-brightest; switch review-recall is **1.00**
  — the injected neighboring-sheet switch is always surfaced for review, and no
  wrong-sheet vertex is confidently accepted.
- **Precision is currently limited on the tested strongly curved real geometry.**
  Thresholds calibrated on flat synthetic sheets over-fire on real papyrus curvature:
  switch precision ~0.19, drift localization remains weak (F1 ~0.01), and ~27% of the
  *clean* surface is flagged for review. On strongly curved real geometry the tool
  currently behaves as a very conservative "flag-for-review" filter rather than a
  precise localizer.

On the tested cube this experiment is a **successful validation of the conservative
safety concept** and a viable expert-in-the-loop workflow — the injected switch is
always surfaced and nothing wrong-sheet is confidently accepted. It also usefully
identifies the next research bottleneck: real curvature increases false positives, so
**curvature-aware residual modelling and improved calibration** are the next
development priorities. This single controlled-corruption cube does not, on its own,
establish general real-scroll precision.

Conclusion: ScrollAnchor is ready for technical community review as an experimental
diagnostic and validation framework. The current real-cube benchmark supports its
conservative safety principle and demonstrates a viable expert-in-the-loop workflow,
while also identifying precision on strongly curved surfaces as the main development
priority. The current release is most useful for assisted review, controlled
benchmark construction, failure analysis, and collaborative method development.
Additional validation with known real annotation failures is needed before
recommending broader or unattended use.

## Current scope and development priorities

These are the current boundaries of what has been demonstrated, and the research
priorities that follow from them:

- **Precise detection is established only for the synthetic benchmark.** Those
  results come from **controlled synthetic** corruptions of a gently curved
  multi-sheet volume, not from real annotation failures.
- **The real-cube experiment uses real CT and real sheet geometry with injected
  corruptions.** It validates the conservative safety concept; it is *not* validation
  on naturally occurring annotation errors, which have **not yet been evaluated**.
- **Real curvature currently increases false positives.** Thresholds tuned on flat
  synthetic sheets over-fire on strongly curved real papyrus (~27% of the clean
  surface flagged on the tested cube); curvature-aware residuals are a priority.
- **Drift localization requires improvement** on real geometry (F1 ~0.01 on the
  tested cube), where genuinely ambiguous zones are surfaced through review rather
  than corrected.
- **Switch detection should become less dependent on a predefined smoothing window.**
  `switch_smooth_window` **must exceed** the switched-patch diameter; too small a
  window silently lowers switch recall.
- **Confidence calibration requires validation on additional cubes.** Normal
  estimation degrades at surface discontinuities; switch detection uses a 3D
  positional residual (not the normal projection) to stay robust to this.
- **Direct VC3D coordinate alignment remains to be verified.** The real-cube surface
  is exported in a **cube-index coordinate frame** (ROI-local indices offset by the
  cube origin) with NRRD metadata validation / axis resolution enforced; full VC3D
  coordinate compatibility is not claimed until visual alignment is checked.

## Development roadmap

Research and engineering directions (priorities, not delivery commitments):

1. Curvature-detrended local residuals.
2. Multi-scale neighboring-sheet-switch detection.
3. Improved confidence calibration on real surfaces.
4. Validation on known naturally occurring failure regions.
5. Direct VC3D coordinate-alignment verification.
6. Integration feedback from annotation and proof-reading workflows.

## Community review and validation

ScrollAnchor is currently seeking technical review from participants familiar with
Vesuvius Challenge surface extraction, annotation, proof-reading, and volumetric
papyrus CT data.

The most useful feedback would include:

- whether the targeted drift and neighboring-sheet-switch failure modes match real
  annotation or tracing problems;
- known small regions containing naturally occurring failures;
- review of the real-cube benchmark construction and coordinate assumptions;
- inspection of generated confidence, drift, switch, and review fields;
- advice on integration with existing VC3D, proof-reading, or segmentation workflows;
- comparison with existing tools that may already address part of the problem.

The current repository should be treated as a working research contribution open to
validation and refinement, not as a community-endorsed solution.

## License

MIT. Interoperates with the Volume Cartographer `tifxyz` format; see
`docs/coordinate_conventions.md` for attribution of format details.

## Author

Olga Ivanova — ivolga.vak@gmail.com
