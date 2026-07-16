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

This is a prototype built for the Vesuvius Challenge 2026 open problems around
label quality, mesh-tracing errors, and sheet-switches (see
`scrollprize.org/2026_open_problems`, problems on *Surface Prediction & Topology*,
*Mesh Tracing Failures*, and *Label Quality & Imprecision*).

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

## Honest limitations

- Validated on **synthetic** corruptions of a controlled multi-sheet volume. Real
  CT (haze in compressed regions, anisotropy, realistic sheet contrast) is not yet
  tested — see the community ask below.
- `switch_smooth_window` **must exceed** the switched-patch diameter; too small a
  window silently lowers switch recall.
- Drift detection precision is diluted by genuinely ambiguous zones (which are, by
  design, surfaced through review rather than corrected).
- Normal estimation degrades at surface discontinuities; switch detection uses a
  3D positional residual (not the normal projection) specifically to be robust to
  this.

## What I'm asking the Scroll Prize community for

The prototype works and is reproducible. To validate on real data I'm looking for
(in order): (1) a few **known problematic tifxyz ROIs** with drift/sheet-switch,
(2) an **expert to inspect** ScrollAnchor's `review_regions.json` on those ROIs,
(3) optionally a **winding/sheet-id volume** to sharpen switch confirmation. If
data can't be shared, I can provide a Docker image and a single run command.

## License

MIT. Interoperates with the Volume Cartographer `tifxyz` format; see
`docs/coordinate_conventions.md` for attribution of format details.

## Author

Olga Ivanova — ivolga.vak@gmail.com
