#!/usr/bin/env python3
"""Real Scroll-1 cube benchmark for ScrollAnchor.

Real CT intensities and real papyrus sheet geometry with controlled surface-label
corruptions (drift + neighbouring-sheet switch). Not validation on naturally
occurring annotation errors.

    python scripts/run_real_cube_benchmark.py --output results/real_cube_02256_02512_04816
    python scripts/run_real_cube_benchmark.py --output ... \
        --volume-nrrd data/real_cube/..._volume.nrrd --mask-nrrd data/real_cube/..._mask.nrrd
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import urllib.request

import numpy as np

from scroll_anchor.config import RunConfig
from scroll_anchor.nrrd_io import header_summary, load_cube
from scroll_anchor.pipeline import analyze_surface
from scroll_anchor.report import apply_review, build_review_regions, write_reports
from scroll_anchor import realcube as rc

CUBE = "02256_02512_04816"
BASE_URL = (
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
    "volumetric-instance-labels/instance-labels-harmonized/" + CUBE
)


def _peak_rss_mib() -> float:
    """Peak resident memory in MiB. ru_maxrss is bytes on macOS, KiB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(rss / scale, 1)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def _resolve_inputs(args) -> tuple:
    if args.volume_nrrd and args.mask_nrrd:
        return args.volume_nrrd, args.mask_nrrd
    data_dir = args.data_dir
    vol = os.path.join(data_dir, f"{CUBE}_volume.nrrd")
    msk = os.path.join(data_dir, f"{CUBE}_mask.nrrd")
    if not args.offline:
        _download(f"{BASE_URL}/{CUBE}_volume.nrrd", vol)
        _download(f"{BASE_URL}/{CUBE}_mask.nrrd", msk)
    for p in (vol, msk):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing NRRD (offline mode): {p}")
    return vol, msk


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def _run_inference(surface, vol_roi, config):
    res = analyze_surface(surface, vol_roi, config)
    apply_review(res.diagnostics, config.review)
    return res


def _harmful_rates(diag, surface, mask_vol, profiles, offsets, source_id):
    """label-as-is / naive-snap / ScrollAnchor harmful acceptance on wrong instance"""
    valid = diag.valid
    pts = surface.points()
    ids_here = np.rint(mask_vol.sample_world(pts, order=0)).astype(np.int64)
    on_wrong = (ids_here != source_id) & (ids_here > 0)

    def rate(accepted):
        acc = accepted & valid
        return float((acc & on_wrong).sum() / max(1, acc.sum()))

    h_asis = rate(valid)
    # naive snap: move to brightest offset along normal, resample instance
    from scroll_anchor.normals import compute_normals
    normals, _ = compute_normals(surface)
    snap = offsets[np.argmax(profiles, axis=2)].astype(np.float32)
    snapped = pts + snap[..., None] * normals
    ids_snap = np.rint(mask_vol.sample_world(snapped, order=0)).astype(np.int64)
    on_wrong_snap = (ids_snap != source_id) & (ids_snap > 0)
    h_naive = float(((valid) & on_wrong_snap).sum() / max(1, valid.sum()))
    h_sa = rate(valid & ~diag.review)
    return {"label_as_is": h_asis, "naive_snap": h_naive, "scroll_anchor": h_sa}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-dir", default="data/real_cube")
    ap.add_argument("--volume-nrrd", default=None)
    ap.add_argument("--mask-nrrd", default=None)
    ap.add_argument("--offline", action="store_true", help="require local NRRD files")
    ap.add_argument("--roi-size", type=int, default=96)
    ap.add_argument("--drift-offset", type=float, default=3.0)
    ap.add_argument("--no-previews", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out = args.output
    src_dir = os.path.join(out, "source")
    os.makedirs(src_dir, exist_ok=True)

    vol_path, mask_path = _resolve_inputs(args)
    vol, mask = load_cube(vol_path, mask_path)
    ct = vol.data.astype(np.float32)
    lab = mask.data.astype(np.int64)

    # --- source metadata ---
    with open(os.path.join(src_dir, "nrrd_headers.json"), "w") as fh:
        json.dump({"volume": header_summary(vol.header), "mask": header_summary(mask.header),
                   "resolved_axis_perm_zyx": list(vol.axis_perm),
                   "spacing_zyx": list(vol.spacing), "origin_zyx": list(vol.origin)}, fh, indent=2)
    report = rc.instance_report(lab)
    with open(os.path.join(src_dir, "cube_summary.json"), "w") as fh:
        json.dump({"cube": CUBE, "shape_zyx": list(ct.shape),
                   "ct_min": float(ct.min()), "ct_max": float(ct.max()),
                   "n_instances": len(report["instances"]),
                   "instances": report["instances"],
                   "adjacent_pairs": report["adjacent_pairs"][:12]}, fh, indent=2)

    # --- ROI + pair selection ---
    sel = rc.select_pair_and_roi(lab, roi_size=args.roi_size)
    z0, y0, x0 = sel.roi_origin
    L = sel.roi_size
    sl = (slice(z0, z0 + L), slice(y0, y0 + L), slice(x0, x0 + L))
    ct_roi = ct[sl]
    mask_roi = lab[sl]
    with open(os.path.join(src_dir, "selected_instances.json"), "w") as fh:
        json.dump({"source_id": sel.source_id, "target_id": sel.target_id,
                   "roi_origin_zyx": list(sel.roi_origin), "roi_size": L,
                   "source_voxels": sel.source_voxels, "target_voxels": sel.target_voxels,
                   "sep_median": sel.sep_median, "sep_p10": sel.sep_p10}, fh, indent=2)

    # --- medial surfaces ---
    ms_s = rc.extract_medial_surface(mask_roi == sel.source_id, sel.roi_origin)
    # extract target on the SAME projection axis so grids align
    depth_t, valid_t = rc.medial_depth_on_grid(mask_roi == sel.target_id, ms_s.proj_axis)
    ms_t = rc.MedialSurface(depth=depth_t, valid=valid_t & rc._largest_component(valid_t),
                            proj_axis=ms_s.proj_axis, roi_origin=sel.roi_origin)

    clean = rc.medial_to_surface(ms_s)
    vol_roi = rc.roi_volume(ct_roi, sel.roi_origin)
    mask_vol = rc.roi_volume(mask_roi, sel.roi_origin)

    os.makedirs(os.path.join(out, "reference_surface"), exist_ok=True)
    from scroll_anchor.tifxyz import write_tifxyz
    write_tifxyz(os.path.join(out, "reference_surface", "surface"), clean, overwrite=True)

    config = RunConfig()
    drift_min = config.diagnostics.drift_min

    # --- corruptions ---
    drift = rc.make_drift(clean, offset=args.drift_offset)
    switch = rc.make_switch(clean, ms_t, mask_roi, sel.target_id, sel.source_id)
    # combined: drift and switch on disjoint patches
    combined = rc.make_drift(clean, offset=args.drift_offset)
    combined_sw = rc.make_switch(combined.surface, ms_t, mask_roi, sel.target_id, sel.source_id)

    # --- inference ---
    res_clean = _run_inference(clean, vol_roi, config)
    res_drift = _run_inference(drift.surface, vol_roi, config)
    res_switch = _run_inference(switch.surface, vol_roi, config)
    res_comb = _run_inference(combined_sw.surface, vol_roi, config)

    for name, surf, res in [("clean_inference", clean, res_clean),
                            ("drift_inference", drift.surface, res_drift),
                            ("switch_inference", switch.surface, res_switch)]:
        regions = build_review_regions(res.diagnostics, config.review)
        write_reports(os.path.join(out, name), surf, res.diagnostics, config, regions)

    # --- metrics ---
    valid = res_clean.diagnostics.valid
    on_wrong_switch = (rc.sample_instance_at(switch.surface, mask_vol) != sel.source_id) & \
                      (rc.sample_instance_at(switch.surface, mask_vol) > 0)

    metrics = {
        "experiment": "real CT intensities + real sheet geometry with controlled corruptions",
        "cube": CUBE,
        "source_id": sel.source_id, "target_id": sel.target_id,
        "roi_origin_zyx": list(sel.roi_origin), "roi_size": L,
        "clean": rc.clean_metrics(res_clean.diagnostics, valid),
        "drift": {
            **rc.drift_metrics(res_drift.diagnostics, drift.region, drift.injected_offset,
                               res_drift.diagnostics.valid, drift_min),
            "injected": drift.info,
        },
        "switch": {
            **rc.switch_metrics(res_switch.diagnostics, switch.region,
                                res_switch.diagnostics.valid, on_wrong_switch),
            "injected": switch.info,
            "harmful_rates": _harmful_rates(res_switch.diagnostics, switch.surface, mask_vol,
                                            res_switch.profiles, res_switch.offsets, sel.source_id),
        },
        "combined_switch": rc.switch_metrics(
            res_comb.diagnostics, combined_sw.region, res_comb.diagnostics.valid,
            (rc.sample_instance_at(combined_sw.surface, mask_vol) != sel.source_id)
            & (rc.sample_instance_at(combined_sw.surface, mask_vol) > 0)),
        "resources": {
            "runtime_seconds": round(time.time() - t0, 2),
            "peak_rss_mb": _peak_rss_mib(),
            "roi_shape_zyx": list(ct_roi.shape),
            "valid_surface_vertices": int(valid.sum()),
            "estimated_spacing_voxels": round(float(res_clean.diagnostics.estimated_spacing), 3),
            "measured_sep_median_voxels": round(sel.sep_median, 3),
            "downloaded_bytes": {
                "volume_nrrd": os.path.getsize(vol_path),
                "mask_nrrd": os.path.getsize(mask_path),
            },
        },
    }
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    if not args.no_previews:
        from scroll_anchor.previews import render_previews
        render_previews(out, ct_roi, mask_roi, sel, ms_s, clean, drift, switch,
                        res_clean, res_drift, res_switch, on_wrong_switch)

    metrics["resources"]["artifact_bytes"] = _dir_size(out)
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps({"source": sel.source_id, "target": sel.target_id,
                      "roi": list(sel.roi_origin), "valid_vertices": int(valid.sum()),
                      "switch_f1": metrics["switch"]["f1"],
                      "drift_f1": metrics["drift"]["f1"],
                      "harmful_sa": metrics["switch"]["harmful_rates"]["scroll_anchor"],
                      "harmful_naive": metrics["switch"]["harmful_rates"]["naive_snap"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
