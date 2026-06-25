#!/usr/bin/env python3
"""
Evaluate a fitted global Ridge model on held-out TSMC combined-patched test cells.

For each held-out cell, loads (N_tasks, 61, 9) input and (N_tasks, 61) output,
predicts y_pred at all 61 voltage points, and computes per-task RMSE (ps) /
NRMSE (%). Aggregates with the same arithmetic mean over tasks that the MAML
validation script uses, then writes:

  <results_dir>/<run_tag>/per_cell_metrics.csv
  <results_dir>/<run_tag>/summary.csv      # mean/median over cells of each bucket

Adaptation modes (--adaptation):
  - none          : raw global ridge prediction at all 61 voltage points.
  - scale_offset  : after the global prediction, per-task affine alignment
                    y' = a * y_global + b fitted on K support points by
                    least squares (TAMEL Stage 1 analog over a ridge base
                    instead of a MAML-pretrained neural net).

Support-set selection follows the MAML validation script:
  --mode interpolation -> indices [0, 13, 30, 45, 60]   (K=5, full range)
  --mode extrapolation -> indices [5, 30, 55]           (K=3, inner range)

Usage:
    # raw global ridge
    python evaluate_ridge_tsmc_combined.py \
        --model_path .../ridge_tsmc_combined_cell_poly2.joblib \
        --topology intra
    # B2: scale_offset adaptation, interpolation support
    python evaluate_ridge_tsmc_combined.py \
        --model_path ... --topology intra \
        --adaptation scale_offset --mode interpolation
"""
import argparse
import csv
import os
import sys
import time

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from test_dataset_config import get_test_config  # noqa: E402


RESULTS_DIR_DEFAULT = (
    "/home/tkdgn2907/Deepsets_test/MAML/Projects/result_management/"
    "iccad2026_rebuttal/regression_baselines"
)


MODE_DEFAULT_INDICES = {
    "interpolation": [0, 13, 30, 45, 60],
    "extrapolation": [5, 30, 55],
}


def per_task_metrics(y_pred, y_true):
    """y_pred, y_true: (N_tasks, 61) in ns. Returns per-task RMSE (ps), NRMSE (%)."""
    # RMSE over the 61 voltage points per task
    rmse_ns = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))  # (N_tasks,)
    rmse_ps = rmse_ns * 1000.0
    y_range = y_true.max(axis=1) - y_true.min(axis=1)
    valid = y_range > 1e-8
    nrmse = np.full_like(rmse_ns, np.nan)
    nrmse[valid] = (rmse_ns[valid] / y_range[valid]) * 100.0
    return rmse_ps, nrmse


def affine_align_per_task(y_pred_global, y_true, support_idx, ridge_eps=1e-6):
    """Per-task least-squares affine alignment.

    Fits a, b per task such that  a * y_pred_global[idx] + b ≈ y_true[idx]
    on the support indices, then applies (a, b) to all 61 voltage points.

    y_pred_global, y_true : (N, 61) float arrays (ns units).
    support_idx           : list/array of voltage indices in [0, 60].
    ridge_eps             : tiny L2 on (a, b) to keep the 2×2 normal eq.
                            invertible when support y is degenerate.
    """
    support_idx = np.asarray(support_idx, dtype=np.int64)
    K = support_idx.shape[0]
    A = y_pred_global[:, support_idx]     # (N, K)  predicted at support
    B = y_true[:, support_idx]            # (N, K)  true       at support

    # Per-task normal equations for [a, b]^T  s.t.  [A, 1] [a; b] ≈ B
    ones = np.ones_like(A)
    X = np.stack([A, ones], axis=-1)      # (N, K, 2)
    XtX = np.einsum('nki,nkj->nij', X, X) # (N, 2, 2)
    Xty = np.einsum('nki,nk->ni', X, B)   # (N, 2)
    XtX += ridge_eps * np.eye(2)[None]    # numerical guard
    coefs = np.linalg.solve(XtX, Xty)     # (N, 2)
    a = coefs[:, 0:1]                     # (N, 1)
    b = coefs[:, 1:2]                     # (N, 1)
    return a * y_pred_global + b          # (N, 61)


def evaluate_cell(pipe, feature_cols, test_in_path, test_out_path,
                  adaptation='none', support_idx=None):
    X3d = torch.load(test_in_path, weights_only=False)
    y2d = torch.load(test_out_path, weights_only=False)
    if y2d.dim() == 3 and y2d.shape[-1] == 1:
        y2d = y2d.squeeze(-1)
    X3d = X3d.numpy()
    y_true = y2d.numpy().astype(np.float32)

    N, T, D = X3d.shape
    X_flat = X3d.reshape(N * T, D)[:, feature_cols].astype(np.float32)
    y_pred_flat = pipe.predict(X_flat)
    y_pred = y_pred_flat.reshape(N, T).astype(np.float32)

    if adaptation == 'scale_offset':
        if support_idx is None or len(support_idx) < 2:
            raise ValueError("scale_offset requires at least 2 support indices")
        y_pred = affine_align_per_task(y_pred, y_true, support_idx).astype(np.float32)

    rmse_ps, nrmse = per_task_metrics(y_pred, y_true)
    return {
        "num_tasks": int(N),
        "rmse_ps_mean": float(np.mean(rmse_ps)),
        "rmse_ps_median": float(np.median(rmse_ps)),
        "nrmse_pct_mean": float(np.nanmean(nrmse)),
        "nrmse_pct_median": float(np.nanmedian(nrmse)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="Path to .joblib produced by fit_ridge_tsmc_combined.py")
    ap.add_argument("--topology", choices=["intra", "agnostic"], required=True,
                    help="intra -> config 6, agnostic -> config 7")
    ap.add_argument("--data_type", choices=["cell", "transition"], default=None,
                    help="Defaults to data_type saved in the joblib")
    ap.add_argument("--cells", type=str, nargs="+", default=None,
                    help="Override list of test cells (default: config defaults)")
    ap.add_argument("--results_dir", type=str, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--run_tag", type=str, default=None,
                    help="Subdir name under results_dir (default: derived from model, topology, adaptation, mode)")
    ap.add_argument("--adaptation", choices=["none", "scale_offset"], default="none",
                    help="Per-task adaptation on top of the global ridge prediction (default: none)")
    ap.add_argument("--mode", choices=["interpolation", "extrapolation"], default="interpolation",
                    help="Support-index preset (only used when --adaptation != none)")
    ap.add_argument("--support_indices", type=int, nargs="+", default=None,
                    help="Override support indices (default: mode-dependent)")
    args = ap.parse_args()

    cfg_id = 6 if args.topology == "intra" else 7
    cfg = get_test_config(cfg_id)
    cells = args.cells if args.cells else cfg["default_cells"]
    test_dir = cfg["test_data_dir"]

    print(f"=== Ridge eval (TSMC combined, {args.topology}, "
          f"{len(cells)} cells) ===")
    print(f"  model_path : {args.model_path}")
    print(f"  cells      : {cells}")

    bundle = joblib.load(args.model_path)
    pipe = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]
    data_type = args.data_type or bundle["data_type"]
    print(f"  data_type  : {data_type} (chosen alpha = {bundle.get('chosen_alpha')})")

    support_idx = (args.support_indices
                   if args.support_indices is not None
                   else MODE_DEFAULT_INDICES[args.mode])
    if args.adaptation != "none":
        print(f"  adaptation : {args.adaptation}   mode={args.mode}   "
              f"support={support_idx}")
    else:
        print(f"  adaptation : none (raw global ridge)")

    tag_suffix = (f"__{args.adaptation}__{args.mode}"
                  if args.adaptation != "none" else "")
    run_tag = args.run_tag or (
        f"{os.path.splitext(os.path.basename(args.model_path))[0]}"
        f"__{args.topology}{tag_suffix}"
    )
    out_dir = os.path.join(args.results_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    t_total = time.time()
    for cell in cells:
        test_in = cfg["test_input_pattern"](cell, data_type).format(test_dir=test_dir)
        test_out = cfg["test_output_pattern"](cell, data_type).format(test_dir=test_dir)
        if not (os.path.exists(test_in) and os.path.exists(test_out)):
            print(f"  [SKIP] {cell}: missing {test_in}")
            continue

        t0 = time.time()
        metrics = evaluate_cell(pipe, feature_cols, test_in, test_out,
                                adaptation=args.adaptation,
                                support_idx=support_idx)
        dt = time.time() - t0
        print(f"  {cell:>22s}  N={metrics['num_tasks']:>6d}  "
              f"RMSE={metrics['rmse_ps_mean']:7.3f} ps  "
              f"NRMSE={metrics['nrmse_pct_mean']:6.3f}%   ({dt:.1f}s)")
        metrics["cell"] = cell
        rows.append(metrics)

    print(f"  total      : {time.time()-t_total:.1f}s")

    per_cell_csv = os.path.join(out_dir, "per_cell_metrics.csv")
    field_order = ["cell", "num_tasks",
                   "rmse_ps_mean", "rmse_ps_median",
                   "nrmse_pct_mean", "nrmse_pct_median"]
    with open(per_cell_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote      : {per_cell_csv}")

    # Aggregate across cells
    if rows:
        summary = {
            "topology": args.topology,
            "data_type": data_type,
            "adaptation": args.adaptation,
            "mode": args.mode if args.adaptation != "none" else "",
            "support_indices": ",".join(str(i) for i in support_idx) if args.adaptation != "none" else "",
            "num_cells": len(rows),
            "rmse_ps_geomean": float(np.exp(np.mean(np.log(
                np.maximum([r["rmse_ps_mean"] for r in rows], 1e-9))))),
            "rmse_ps_mean_of_means": float(np.mean([r["rmse_ps_mean"] for r in rows])),
            "rmse_ps_median_of_means": float(np.median([r["rmse_ps_mean"] for r in rows])),
            "nrmse_pct_geomean": float(np.exp(np.mean(np.log(
                np.maximum([r["nrmse_pct_mean"] for r in rows], 1e-9))))),
            "nrmse_pct_mean_of_means": float(np.mean([r["nrmse_pct_mean"] for r in rows])),
        }
        summary_csv = os.path.join(out_dir, "summary.csv")
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader()
            w.writerow(summary)
        print(f"  wrote      : {summary_csv}")
        print()
        print("  === SUMMARY ===")
        for k, v in summary.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
