# PHercParis4 w110-112 exploratory render review

Exploratory 2D render-anomaly review for the PHercParis4 segment w110-112 surface
render.

## Artifacts

- [report.pdf](report.pdf) - multi-page review packet
- [overlay.png](overlay.png) - ranked candidate boxes over the whole render
- [top_candidates.png](top_candidates.png) - contact sheet of the top-ranked crops
- `summary.json`, `metadata.json`, `regions.json` - counts, parameters, and per-region data

## Summary

The complete downsampled JPG was scanned at half its linear resolution, so this is
a whole-render pass and not a partial crop.

- 200 ranked candidates were exported from 428 components that passed the
  minimum-size filter
- 228 additional passing candidates were not exported because of the configured cap
- Candidates are exploratory 2D visual anomalies, not confirmed sheet skips
- Mapped full-render coordinates use the documented factor of 8, and are not
  verified VC3D coordinates

The source JPG and TIFF are not stored in this repository

## Regenerate the report

Report-only regeneration for this run (see the [main README](../../README.md) for
how the tool works):

```bash
scroll-anchor render-report \
  --results results/pherc-render \
  --render path/to/PHercParis4-20260623163339-2.4um-0.22m-78keV-volume-20260411134726-alpha-overlay-combined-ds8.jpg
```

## Attribution

Source render derived from Vesuvius Challenge open data (PHercParis4 segment
20260623163339-w110-112). This review does not imply endorsement by the Vesuvius
Challenge.

- Author: Olga Ivanova
- Repository: https://github.com/olgaiv39/scroll-anchor
