# CPU median diagnostics

The surface-switch residual previously ran three SciPy median filters. The
finite float32/float64 path now uses tiled NumPy selection with the same odd
window and nearest-edge padding. Non-finite values and unsupported types retain
SciPy behavior. The neighborhood workspace is capped at 16 MiB; input, padding
and output memory are additional.

On macOS arm64 with Python 3.12.14, NumPy 2.5.3 and SciPy 1.18.1, the median of
nine paired full-pipeline speedups was **1.67×**, with individual cases between
1.32× and 1.83×. Each case used one warmup followed by five alternating paired
runs of `analyze_surface`, without previews or disk export. See `results.json`
for every timing, dataset identity and crop. These are measurements from one
machine, not a speed guarantee.

All diagnostic array fields were byte-identical in every recorded run. The
canonical tests separately compare numeric median results against SciPy using
irregular data, tile boundaries, edges and non-finite input. Signed-zero ties
can choose different median representatives, so universal byte equality of the
median kernel itself is not claimed.

## Reproduce

From a checkout containing this change, install the project's `dev` and
`benchmark` extras in your chosen environment. The two scripts below add no
runtime dependency to ScrollAnchor.

```sh
python scripts/download_median_benchmark.py \
  --manifest benchmarks/median/data-manifest.json --output data/cpu-cubes
git show ad29796d23785e9fb2220783c929172c76e783d1:src/scroll_anchor/diagnostics.py > /tmp/scroll-anchor-before.py
python scripts/benchmark_median.py \
  --baseline-source /tmp/scroll-anchor-before.py \
  --data-dir data/cpu-cubes --output /tmp/median-results.json
python -m pytest -q
```

The downloader verifies the pinned SHA-256 hashes before accepting files. The
full compressed set is about 48 MB. All three cubes are from PHercParis4.
`02256_02512_04816` is the development cube; `00064_02664_04304` and
`01744_02000_04816` were reserved before development. Holdout pair/crop selection
uses instance geometry only. Each case uses a 96³ ROI and a clean surface or a
controlled wrong-sheet assignment with half-width 8 or 32 grid vertices.

## Limits and attribution

This accelerates the current detector. It does not fix its documented inability
to reliably detect wrong-sheet regions larger than the median window. Reference
surfaces are extracted instance-label medial surfaces, not independently checked
writing surfaces. This benchmark does not establish natural-error accuracy,
readable letters, model quality, cross-scroll performance or a Grand Prize result.

The implementation and benchmark build on Olga Ivanova's MIT-licensed
ScrollAnchor, including its real-cube fixture and corruption machinery. Data is
Vesuvius Challenge PHercParis4 CT and harmonized volumetric instance labels from
the URLs in `data-manifest.json`. Data and derived visualizations retain their
source terms; the software license does not relicense them. See the
[official data attribution and license guidance](https://scrollprize.org/data).

AI-assisted change. Validation here was run by an agent on real scroll data;
it is not evidence of independent human use or maintainer acceptance.
