"""Paired full-pipeline timing and byte equality on fixed real-data cubes."""
import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import scipy
import scroll_anchor.diagnostics as diagnostics
from scroll_anchor import realcube as rc
from scroll_anchor.config import RunConfig
from scroll_anchor.pipeline import analyze_surface
from scroll_anchor.nrrd_io import load_cube


def prepare(cube, holdout=False, data_dir=None):
    folder = (Path(data_dir) if data_dir else Path("data/cpu-cubes")) / cube
    ct, labels = load_cube(str(folder/f"{cube}_volume.nrrd"), str(folder/f"{cube}_mask.nrrd"))
    if not holdout:
        origin, sid, tid = (154,160,90), 328, 329
    else:
        # Geometry-only selection from a bounded central region. Never inspect
        # detector outcomes to select a pair or an evaluation crop.
        central = labels.data[64:192,64:192,64:192]
        selection = rc.select_pair_and_roi(central, roi_size=96)
        origin = tuple(v+64 for v in selection.roi_origin)
        sid, tid = selection.source_id, selection.target_id
    slices = tuple(slice(v, v+96) for v in origin)
    mask = labels.data[slices]
    vol = rc.roi_volume(ct.data[slices], origin)
    source = rc.extract_medial_surface(mask == sid, origin)
    depth, valid = rc.medial_depth_on_grid(mask == tid, source.proj_axis)
    target = rc.MedialSurface(depth, valid & rc._largest_component(valid), source.proj_axis, origin)
    return rc.medial_to_surface(source), target, mask, vol, sid, tid, origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "scroll_anchor._baseline_diagnostics", args.baseline_source,
    )
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    implementations = {"baseline": old._robust_residual_magnitude,
                       "fixed": diagnostics._robust_residual_magnitude}
    cases = []
    cfg = RunConfig()
    for cube, holdout in [("02256_02512_04816", False),
                          ("00064_02664_04304", True),
                          ("01744_02000_04816", True)]:
        clean, target, mask, volume, sid, tid, origin = prepare(
            cube, holdout, args.data_dir,
        )
        for half in (0, 8, 32):
            corruption = rc.make_switch(clean, target, mask, tid, sid, half=half)
            timings = {name: [] for name in implementations}
            reference = None
            for repeat in range(args.repeats + 1):
                order = ("baseline", "fixed") if repeat % 2 == 0 else ("fixed", "baseline")
                for name in order:
                    diagnostics._robust_residual_magnitude = implementations[name]
                    started = time.perf_counter()
                    result = analyze_surface(corruption.surface, volume, cfg)
                    elapsed = time.perf_counter() - started
                    arrays = {field.name: getattr(result.diagnostics, field.name)
                              for field in dataclasses.fields(result.diagnostics)}
                    if reference is None:
                        reference = arrays
                    for key, actual in arrays.items():
                        expected = reference[key]
                        if isinstance(actual, np.ndarray):
                            assert actual.dtype == expected.dtype and actual.shape == expected.shape
                            assert actual.tobytes() == expected.tobytes(), (cube, half, name, key)
                        else:
                            assert actual == expected, (cube, half, name, key)
                    if repeat:
                        timings[name].append(elapsed)
            medians = {name: statistics.median(times) for name, times in timings.items()}
            record = {"cube": cube, "holdout": holdout, "half": half,
                      "origin_zyx": origin, "source_id": sid, "target_id": tid,
                      "valid_vertices": int(result.diagnostics.valid.sum()),
                      "diagnostic_fields_byte_identical": True,
                      "timings_seconds": timings, "median_seconds": medians,
                      "speedup": medians["baseline"] / medians["fixed"]}
            cases.append(record)
            print(json.dumps(record), flush=True)
    diagnostics._robust_residual_magnitude = implementations["fixed"]
    receipt = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
               "machine": platform.machine(), "platform": platform.platform(),
               "method": "one warmup then alternating paired runs; full analyze_surface; no previews or disk export",
               "cases": cases,
               "aggregate_median_speedup": statistics.median(case["speedup"] for case in cases),
               "scope": "three cubes from PHercParis4, 96^3 ROIs; no training or natural-error accuracy claim"}
    Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
