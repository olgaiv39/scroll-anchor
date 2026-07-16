# Method

ScrollAnchor scores an approximate `tifxyz` surface against CT evidence. It is
non-neural and needs no training or ground truth at inference.

## Pipeline

1. **Normals** (`normals.py`): estimate per-vertex world normals from the quad
   grid; flip to a consistent hemisphere.
2. **Profile sampling** (`sampling.py`): for each vertex, sample the CT intensity
   along the normal over `±radius` voxels at `step` spacing (trilinear, CPU).
   Processing is chunked over grid rows to bound peak memory
   (`chunk_rows * W * T` floats at a time).
3. **Diagnostics** (`diagnostics.py`):
   - **Peaks**: `scipy.signal.find_peaks` on each normalized profile; a
     distance-weighted score `height * exp(-|offset|/spacing)` picks the *chosen*
     sheet peak and yields a best-vs-second **margin**.
   - **Drift**: `chosen_offset` = signed distance to the chosen peak; a drifted
     surface shows a consistent nonzero offset back toward the true sheet.
   - **Sheet-switch**: a **robust** (median-consensus over a large window) 3D
     positional residual of ~one sheet spacing, confirmed by high on-sheet
     evidence (a switch sits on a *real, wrong* sheet). **Hysteresis** grows
     strong cores across the whole switched patch (recovering the boundary ring).
     The 3D magnitude is used (not the normal projection) because normals are
     unreliable at the switch cliff.
   - **Sheet spacing** is estimated from median inter-peak distance if not set.
   - **Confidence** = `contrast * margin_conf * geom_conf * evidence`, all in
     [0, 1]; the product makes the score conservative.
   - **Review** = switch, or `confidence < threshold`, or large drift.
4. **Correction (optional, off by default)**: propose moving to the chosen peak
   only when confidence and margin are high, the move is small, and there is no
   switch. Otherwise the vertex is flagged, not moved.

## Why "harmful acceptance rate" is the primary metric

The worst outcome is a confident label on the **wrong sheet**. We therefore score
each method by the fraction of vertices it *accepts* (keeps or moves) that land on
the wrong sheet. A tool that flags ambiguous regions for review - rather than
guessing - drives this toward zero even at the cost of leaving some regions
unresolved. That trade-off is the entire point.
