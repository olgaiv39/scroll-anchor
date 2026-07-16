---
name: validate-real-cube
description: Run the ScrollAnchor real-cube benchmark on a Scroll instance-label NRRD cube (real CT + real sheet geometry, controlled drift/switch corruptions). Use when validating ScrollAnchor on a real cube, adding a new cube, or checking whether results transfer to real data.
---

# validate-real-cube

Runs and verifies the repeatable real-cube diagnostic workflow. Real CT intensities
and real papyrus sheet geometry with **controlled, injected** corruptions — never
describe results as validation on naturally occurring annotation errors.

## Procedure

1. **Input validation**
   - Ensure `pip install -e ".[benchmark]"` (pynrrd, matplotlib) is available.
   - Offline: pass `--volume-nrrd`/`--mask-nrrd` or place files in `data/real_cube/`
     and use `--offline`. Online: the script downloads by cube id.

2. **Load + resolve axes** (`scroll_anchor.nrrd_io.load_cube`)
   - Confirm resolved `axis_perm`, `spacing`, `origin`; internal order is `[z,y,x]`.
   - Verify CT/mask share shape, spacing, origin (loader raises otherwise).
   - STOP if orientation is ambiguous or CT/mask alignment cannot be verified.

3. **ROI + instance selection** (`realcube.select_pair_and_roi`)
   - Picks the source/target pair maximising reliable switch-constructible area.
   - STOP and report if no neighbouring pair yields enough valid support.

4. **Medial-surface extraction** (`realcube.extract_medial_surface`)
   - Projection axis chosen from geometry; single-run columns only; largest CC kept.
   - STOP if no reliable structured surface can be extracted.

5. **Controlled corruptions**
   - Drift: `make_drift` moves a compact patch along real normals by a known offset.
   - Switch: `make_switch` replaces a patch with the neighbour's medial surface and
     verifies each target coordinate lands on the target instance.

6. **Inference** — correction disabled (primary diagnostic). Run clean, drift,
   switch, combined surfaces through `analyze_surface`.

7. **Metrics** (`metrics.json`) — clean review/stability, drift P/R/F1 + MAE + sign,
   switch P/R/F1 + review-recall + harmful acceptance, resources.

8. **Previews** (`previews/*.png`) — CT slices with instance contours, reference
   surface, corruption patches, confidence/drift/switch/review, TP/FP/FN. The
   construction must be visually inspectable.

9. **Stop-condition check** — report honestly (a negative result is acceptable) if:
   axis order unresolved; CT/mask misaligned; no reliable surface/neighbour;
   clean surface broadly flagged; intensity assumptions do not transfer; success
   would require redesigning the algorithm; construction not visually verifiable.

## One command

```bash
python scripts/run_real_cube_benchmark.py --output results/real_cube_<CUBE> [--offline]
```

## Interpreting output
- Safety = harmful acceptance near 0 and switch review-recall near 1.
- Precision = clean review fraction low and switch/drift precision high.
- On tightly curved real sheets, expect strong safety but weak precision; do not
  tune thresholds to force a positive result.
