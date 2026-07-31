"""ScrollAnchor command-line interface"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

from .config import RunConfig
from .logging_setup import configure, get_logger
from .pipeline import analyze_surface
from .report import apply_review, build_review_regions, write_reports
from .tifxyz import Surface, read_tifxyz
from .volume import VolumeROI, load_zarr_roi, open_zarr

log = get_logger(__name__)


def _load_config(path: Optional[str]) -> RunConfig:
    if path:
        return RunConfig.from_yaml(path)
    return RunConfig()


def _load_volume_for_surface(volume_path: str, surface: Surface, radius: float) -> VolumeROI:
    """Load a CT volume ROI. Supports .npy (in-memory) or a zarr path/URL."""
    if volume_path.endswith(".npy"):
        arr = np.load(volume_path)
        return VolumeROI.from_array(arr, origin=(0, 0, 0))
    pts = surface.points()[surface.valid]
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    margin = int(np.ceil(radius)) + 2
    array = open_zarr(volume_path)
    return load_zarr_roi(
        array,
        ((int(xmin), int(xmax)), (int(ymin), int(ymax)), (int(zmin), int(zmax))),
        margin=margin,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.enable_correction:
        config.correction.enabled = True
    surface = read_tifxyz(args.surface)
    volume = _load_volume_for_surface(args.volume, surface, config.sampling.radius)
    result = analyze_surface(surface, volume, config)
    diag = result.diagnostics
    apply_review(diag, config.review)
    regions = build_review_regions(diag, config.review)
    write_reports(args.output, surface, diag, config, regions, write_channels=not args.no_channels)
    log.info(
        "wrote reports to %s (%d review regions, %d correction proposals)",
        args.output, len(regions), int(np.sum(np.isfinite(diag.correction_offset))),
    )
    return 0


def cmd_analyze_render(args: argparse.Namespace) -> int:
    # Exploratory 2D render analysis. Separate from the 3D analyze pipeline: it uses
    # only a downsampled 2D JPG and reports candidate visual discontinuities, NOT
    # confirmed sheet switches, 3D drift, or voxel displacement.
    from .render2d import RenderParams, analyze_render

    params = RenderParams(
        working_downsample=args.working_downsample,
        max_working_pixels=args.max_pixels,
        jpg_to_full_factor=args.full_render_factor,
    )
    summary = analyze_render(
        args.render, args.output, params,
        tifxyz_dir=args.tifxyz_dir,
        render_background_threshold=args.render_background_threshold,
        render_background_min_component_pixels=args.render_background_min_component_pixels,
    )
    log.info(
        "render analysis: %d exported / %d above-threshold / %d total response(s), "
        "score %.3f-%.3f, processed %s, %.1fs -> %s",
        summary["n_regions_exported"], summary["n_regions_above_threshold"],
        summary["n_regions_total"], summary["exported_score_range"][0],
        summary["exported_score_range"][1], summary["processed_shape_rowcol"],
        summary["runtime_seconds"], args.output,
    )
    if summary["tifxyz_valid_mask_used"]:
        log.info("tifxyz valid-surface mask applied as diagnostics only; "
                 "scores and ranking are unchanged")
    if summary["render_background_mask_used"]:
        log.info("render-derived background diagnostics applied (grayscale threshold "
                 "%d, min component %d processed px); scores and ranking are unchanged",
                 args.render_background_threshold,
                 args.render_background_min_component_pixels)
    return 0


def cmd_render_report(args: argparse.Namespace) -> int:
    # Report-only: rebuild report.pdf (and top_candidates.png if the render is
    # available) from an existing results directory. Does NOT run the detector,
    # read diagnostics.npz, or modify the JSON/NPZ/overlay artifacts.
    from .render_report import build_report

    info = build_report(args.results, args.render)
    counts = info["counts"]
    log.info(
        "render report: %d-page PDF, %d ranked candidates, crops from render=%s -> %s",
        info["n_pages"], info["n_regions"], info["used_render"], info["report"],
    )
    if not info["used_render"]:
        log.info("source render unavailable; reused existing top_candidates.png "
                 "(counts: exported=%s of %s passing)",
                 counts.get("n_regions_exported"), counts.get("n_regions_above_threshold"))
    return 0


def cmd_map_render_candidates(args: argparse.Namespace) -> int:
    # Mapping stage only: exported candidate -> source-render location -> tifxyz
    # cell -> x/y/z values. Does NOT rerun detection, change scores or ranking,
    # open a CT volume, or convert coordinates to a CT array index.
    from .tifxyz_mapping import map_render_candidates

    if os.path.isdir(args.output) and os.listdir(args.output):
        log.info("output directory already exists and is not empty: %s "
                 "(existing files with the same names will be overwritten)", args.output)

    info = map_render_candidates(
        args.regions, args.metadata, args.tifxyz_dir, args.output,
        ranks=args.rank, component_ids=args.component_id,
        assume_identity_orientation=args.assume_identity_orientation,
        render_path=args.render,
    )
    log.info(
        "mapped %d candidate(s) onto tifxyz %s (source scale %s, processed scale %s) -> %s",
        info["n_selected"], info["tifxyz_shape_rowcol"],
        info["source_to_tifxyz_rowcol"], info["processed_to_tifxyz_rowcol"], args.output,
    )
    for e in info["candidates"]:
        log.info(
            "  rank %d = component %d: tifxyz (row %d, col %d) valid=%s x=%s y=%s z=%s",
            e["exported_rank"], e["component_id"], e["tifxyz_row"], e["tifxyz_col"],
            e["tifxyz_cell_valid"], e["x"], e["y"], e["z"],
        )
    log.info("identity orientation was assumed on request and is NOT independently "
             "verified; x/y/z are raw tifxyz values, not CT array indices, and CT "
             "volume axis order and bounds remain unverified")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .synth import make_scene
    from .metrics import evaluate

    config = _load_config(args.config)
    config.seed = args.seed
    if args.enable_correction:
        config.correction.enabled = True

    scene = make_scene(H=args.size, W=args.size, seed=args.seed)
    result = analyze_surface(scene.corrupt, scene.volume, config)
    diag = result.diagnostics
    apply_review(diag, config.review)
    regions = build_review_regions(diag, config.review)

    bench = evaluate(
        diag, scene.gt, scene.sheet_model, scene.corrupt.points(), result.normals,
        result.profiles, result.offsets, config.diagnostics.drift_min,
    )
    os.makedirs(args.output, exist_ok=True)
    write_reports(args.output, scene.corrupt, diag, config, regions)
    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(bench.to_dict(), fh, indent=2)

    b = bench.to_dict()
    log.info("=== ScrollAnchor synthetic benchmark ===")
    log.info("switch  P/R/F1: %.3f / %.3f / %.3f",
             b["switch_detection"]["precision"], b["switch_detection"]["recall"],
             b["switch_detection"]["f1"])
    log.info("drift   P/R/F1: %.3f / %.3f / %.3f",
             b["drift_detection"]["precision"], b["drift_detection"]["recall"],
             b["drift_detection"]["f1"])
    log.info("drift displacement MAE (voxels): %.3f", b["drift_displacement_mae"])
    log.info("harmful acceptance rate  label-as-is: %.3f", b["harmful_rate_label_as_is"])
    log.info("harmful acceptance rate  naive-snap : %.3f", b["harmful_rate_naive_snap"])
    log.info("harmful acceptance rate  ScrollAnchor: %.3f", b["harmful_rate_scroll_anchor"])
    log.info("clean stability: %.3f   accepted frac: %.3f   review frac: %.3f",
             b["clean_stability"], b["accepted_frac_scroll_anchor"], b["review_frac"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scroll-anchor", description=__doc__)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Analyze a tifxyz surface against a CT volume/ROI")
    a.add_argument("--surface", required=True, help="tifxyz surface directory")
    a.add_argument("--volume", required=True, help=".npy volume or zarr path/URL [z,y,x]")
    a.add_argument("--config", default=None, help="YAML config (defaults if omitted)")
    a.add_argument("--output", required=True, help="output directory")
    a.add_argument("--enable-correction", action="store_true", help="propose conservative moves")
    a.add_argument("--no-channels", action="store_true", help="skip writing tifxyz channels")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser(
        "analyze-render",
        help="Exploratory 2D analysis of a downsampled surface render (JPG); flags "
             "candidate visual discontinuities only, not confirmed sheet switches",
    )
    r.add_argument("--render", required=True, help="downsampled 2D grayscale render (JPG)")
    r.add_argument("--output", required=True, help="output directory")
    r.add_argument("--working-downsample", type=int, default=2,
                   help="additional downsample applied to the JPG before analysis")
    r.add_argument("--max-pixels", type=int, default=60_000_000,
                   help="safety cap on processed pixel count")
    r.add_argument("--full-render-factor", type=int, default=8,
                   help="documented JPG->full-render coordinate factor (mapped, not verified)")
    r.add_argument("--tifxyz-dir", default=None,
                   help="optional tifxyz directory (x.tif/y.tif/z.tif) matching the render; "
                        "derives a valid-surface mask and adds mesh-boundary diagnostics to "
                        "each candidate. Does not change scores or ranking")
    r.add_argument("--render-background-threshold", type=int, default=None,
                   help="optional 8-bit grayscale level (0..255); pixels at or below it "
                        "count as near-black. Must be given together with "
                        "--render-background-min-component-pixels. No default: the "
                        "value is dataset- and resolution-dependent")
    r.add_argument("--render-background-min-component-pixels", type=int, default=None,
                   help="optional minimum size, in processed working-resolution pixels, "
                        "of an edge-connected near-black component counted as render "
                        "background. Adds descriptive fields to each candidate; does not "
                        "change scores or ranking")
    r.set_defaults(func=cmd_analyze_render)

    rr = sub.add_parser(
        "render-report",
        help="Rebuild the review PDF from an existing render-analysis results "
             "directory (report-only; does not run the detector)",
    )
    rr.add_argument("--results", required=True,
                    help="results directory holding metadata/summary/regions/overlay")
    rr.add_argument("--render", default=None,
                    help="optional source JPG for higher-quality crop pages")
    rr.set_defaults(func=cmd_render_report)

    mc = sub.add_parser(
        "map-render-candidates",
        help="Map already exported 2D render candidates onto a compatible tifxyz "
             "raster and read x/y/z at the containing cell (mapping only; does not "
             "rerun detection, change scores/ranking, or open a CT volume)",
    )
    mc.add_argument("--regions", required=True,
                    help="regions.json from an analyze-render run (read-only)")
    mc.add_argument("--metadata", required=True,
                    help="metadata.json from the same run (read-only)")
    mc.add_argument("--tifxyz-dir", required=True,
                    help="tifxyz directory holding x.tif/y.tif/z.tif")
    mc.add_argument("--output", required=True, help="output directory")
    mc.add_argument("--rank", type=int, action="append", default=None,
                    help="exported rank to map (1-based position in the ranked "
                         "subset, NOT the component ID); repeatable")
    mc.add_argument("--component-id", type=int, action="append", default=None,
                    help="connected-component ID to map (NOT the exported rank); "
                         "repeatable. At least one --rank or --component-id required")
    mc.add_argument("--render", default=None,
                    help="optional source JPG; when given, writes a "
                         "render-to-tifxyz mapping preview PNG (not a CT overlay)")
    mc.add_argument("--assume-identity-orientation", action="store_true",
                    help="required acknowledgement that the tifxyz raster shares the "
                         "render's origin, orientation and extent (no flip, transpose, "
                         "crop or offset). The metadata does not record orientation, so "
                         "it is never assumed silently. Flips are not implemented")
    mc.set_defaults(func=cmd_map_render_candidates)

    b = sub.add_parser("benchmark", help="Run the synthetic corruption benchmark")
    b.add_argument("--output", required=True)
    b.add_argument("--config", default=None)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--size", type=int, default=80)
    b.add_argument("--enable-correction", action="store_true")
    b.set_defaults(func=cmd_benchmark)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
