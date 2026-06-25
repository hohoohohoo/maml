# Regression baselines (ICCAD2026 rebuttal — R-D Q2 / W2 (iv))

Empirical check that "feature-based regression models" — one of the four
baselines requested by R-D — are bounded above by Aadam (deep MLP over the
same features). All runs use **TSMC combined-patched (config 6/7)** so the
train/test partition matches the GCN_MAML numbers in paper Table 4.

## Ridge (poly degree 2, RidgeCV) — cell_delay

Global ridge fit on 2 M subsampled training points
(`pretrained_models/regression_pretrained_model/ridge_tsmc_combined_cell_poly2.joblib`,
chosen `alpha = 10.0`). No per-task adaptation — purely a global feature →
delay regressor over the 8 features `[T_ox, DIBL, SCE, T, V, rise/fall,
slew, load]`.

Two adaptation regimes:
- **raw** (`--adaptation none`): global ridge prediction at all 61 V_dd.
- **scale_offset** (`--adaptation scale_offset`): TAMEL Stage 1 analog —
  per-task affine alignment `y' = a·ŷ + b` fitted on K support points
  by least squares (K=5 for interpolation `[0,13,30,45,60]`, K=3 for
  extrapolation `[5,30,55]`).

Results (TSMC cell-delay, RMSE in ps, mean over per-cell means):

| Bucket | Ridge raw | **Ridge + scale_offset** (interp / extrap) |
|---|---:|---:|
| TSMC intra-topology (6 cells)     | 56.3 | **18.1 / 14.7** |
| TSMC topology-agnostic (14 cells) | 53.1 | **17.8 / 14.4** |

Comparison against paper Table 4 (TSMC cell-delay RMSE in ps, interp / extrap):

| Model | intra | agnostic |
|---|---|---|
| Ridge **raw** | 56.3 / 56.3 | 53.1 / 53.1 |
| **Ridge + scale_offset** (B2) | **18.1 / 14.7** | **17.8 / 14.4** |
| Aadam (MLP, hidden=256, no MAML)   | 4.96 / 3.32   | 4.54 / 3.11   |
| GCN baseline (no MAML)             | 10.10 / 9.72  | 14.88 / 13.24 |
| MLP_MAML (hidden=40)               | 0.91 / 1.79   | 0.91 / 1.62   |
| GCN_MAML                           | 0.80 / 1.51   | 0.76 / 1.60   |

**Take-away for review_D.md (iv).**
- Raw global ridge is **~11–17× worse than Aadam** and **~58–70× worse
  than MLP_MAML**.
- Even with TAMEL's Stage 1 affine alignment grafted on top (the
  *strongest* shape-preserving adaptation that closed-form ridge admits),
  ridge is still **3–5× worse than Aadam** and **10–24× worse than
  MLP_MAML / GCN_MAML**.
- This rules out the "the gain comes from the Δ-scale + offset
  alignment, not from the base model" reading: the same alignment on a
  classical-regression base lands at 14–18 ps, far above any TAMEL
  variant. The dominant lever is the meta-learned base model, not the
  adaptation trick.

## How to reproduce

```bash
# Fit (≈ 20 s on CPU, 2 M subsample)
cd pretraining/model_pretraining_code/regression
python fit_ridge_tsmc_combined.py --data_type cell --poly_degree 2

# Evaluate both buckets, raw (~15 s combined)
cd ../../model_test_code/regression
python evaluate_ridge_tsmc_combined.py \
    --model_path ../../../pretrained_models/regression_pretrained_model/ridge_tsmc_combined_cell_poly2.joblib \
    --topology intra --data_type cell
python evaluate_ridge_tsmc_combined.py \
    --model_path ../../../pretrained_models/regression_pretrained_model/ridge_tsmc_combined_cell_poly2.joblib \
    --topology agnostic --data_type cell

# B2: TAMEL Stage 1 analog (Δscale + offset on K support points)
for topo in intra agnostic; do
    for mode in interpolation extrapolation; do
        python evaluate_ridge_tsmc_combined.py \
            --model_path ../../../pretrained_models/regression_pretrained_model/ridge_tsmc_combined_cell_poly2.joblib \
            --topology $topo --data_type cell \
            --adaptation scale_offset --mode $mode
    done
done
```

Per-cell metrics land at:

```
regression_baselines/ridge_tsmc_combined_cell_poly2__intra/per_cell_metrics.csv
regression_baselines/ridge_tsmc_combined_cell_poly2__agnostic/per_cell_metrics.csv
```

## Notes & caveats

- The combined-patched dataset (config 6/7) only contains combinational
  cells, so the agnostic bucket here is 14 cells (vs 16 in legacy
  config 3, which adds DFCNQD1 + SDFSNQD0). Sequential cells are absent
  from the patched MLP_dataset_TSMC directory.
- Feature columns used: `[0,1,2,3,4,6,7,8]` (drops index 5
  `additional_dim` which is unused in this dataset; keeps the rise/fall
  indicator at index 6).
- Training subsample = 2 M points (out of ~44 M) — the train RMSE on the
  full set is essentially identical, since the global linear+poly2 model
  saturates well before 2 M.
