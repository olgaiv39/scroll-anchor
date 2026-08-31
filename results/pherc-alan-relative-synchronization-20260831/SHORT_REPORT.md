# Alan external relative-synchronization evaluation

The 20 external PHerc cases were prepared and shared by ScrollPrize community contributor altommo. They include genuine surfaces, controlled normal shifts, and synthetic sheet-switch cases

## Evaluation

ScrollAnchor relative synchronization was evaluated with the threshold-free `PHASE_PATCH_SCORE`

- AUROC: 0.672
- Average Precision: 0.385
- Sheet-switch ranks: 3, 4, 8, 12

## Provenance

Scores were originally frozen before truth reveal. A later geometry review found one section-local scoring implementation bug affecting one control case. Correcting it did not change AUROC, Average Precision, or the sheet-switch ranks

## Interpretation

The result provides evidence of meaningful correspondence structure, but it is not a finished sheet-switch classifier
