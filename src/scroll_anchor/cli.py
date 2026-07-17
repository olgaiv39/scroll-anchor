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
    summary = analyze_render(args.render, args.output, params)
    log.info(
        "render analysis: %d candidate region(s), processed %s, %.1fs -> %s",
        summary["n_regions"], summary["processed_shape_rowcol"],
        summary["runtime_seconds"], args.output,
    )
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
    r.set_defaults(func=cmd_analyze_render)

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
