#!/usr/bin/env python3
"""
Fit a global Ridge regression on TSMC combined-patched (config 6/7) training data.

Compared to TAMEL's per-task adaptation, this is a single global model:
  X_global = (process+condition+voltage) -> y_global = delay
with no per-task adaptation. Saves the fitted sklearn estimator (joblib) to
pretrained_models/regression_pretrained_model/.

Usage:
    python fit_ridge_tsmc_combined.py --data_type cell
    python fit_ridge_tsmc_combined.py --data_type transition --poly_degree 2
"""
import argparse
import os
import time

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


TRAIN_INPUT_TMPL = (
    "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/MLP_dataset_TSMC/"
    "combined_data/tsmc_topology_agnostic_train_input_{dt}.pth"
)
TRAIN_OUTPUT_TMPL = (
    "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/MLP_dataset_TSMC/"
    "combined_data/tsmc_topology_agnostic_train_output_{dt}.pth"
)
MODEL_DIR = (
    "/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/"
    "regression_pretrained_model"
)

# 9-D feature layout (per MAML/MLP training):
#   0:a_param  1:b_param  2:c_param  3:temperature  4:voltage
#   5:additional_dim  6:delay_indicator  7:slew  8:load_cap
DEFAULT_FEATURE_COLS = [0, 1, 2, 3, 4, 6, 7, 8]  # drop col 5 (additional_dim, unused)


def load_train_tensors(data_type):
    inp = torch.load(TRAIN_INPUT_TMPL.format(dt=data_type), weights_only=False)
    out = torch.load(TRAIN_OUTPUT_TMPL.format(dt=data_type), weights_only=False)
    if out.dim() == 3 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return inp.numpy(), out.numpy()


def flatten_dataset(X3d, y2d, feature_cols, subsample=None, rng=None):
    """(N_tasks, 61, 9) + (N_tasks, 61) -> (N_tasks*61, len(feature_cols)), (N_tasks*61,)"""
    N, T, D = X3d.shape
    X = X3d.reshape(N * T, D)[:, feature_cols].astype(np.float32)
    y = y2d.reshape(N * T).astype(np.float32)
    if subsample is not None and subsample < X.shape[0]:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=subsample, replace=False)
        X = X[idx]
        y = y[idx]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_type", choices=["cell", "transition"], default="cell")
    ap.add_argument("--poly_degree", type=int, default=2,
                    help="Polynomial feature degree (1=linear ridge, 2=interactions+squares)")
    ap.add_argument("--subsample", type=int, default=2_000_000,
                    help="Subsample N rows for fitting (None=all). 2M is usually enough for ridge.")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.01, 0.1, 1.0, 10.0, 100.0],
                    help="RidgeCV alpha grid")
    ap.add_argument("--feature_cols", type=int, nargs="+", default=None,
                    help=f"Feature column indices (default: {DEFAULT_FEATURE_COLS})")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="Optional suffix on saved model filename")
    args = ap.parse_args()

    feature_cols = args.feature_cols if args.feature_cols else DEFAULT_FEATURE_COLS
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"=== Ridge fit (TSMC combined, data_type={args.data_type}) ===")
    print(f"  feature_cols   : {feature_cols}")
    print(f"  poly_degree    : {args.poly_degree}")
    print(f"  subsample      : {args.subsample}")
    print(f"  alphas (CV)    : {args.alphas}")

    t0 = time.time()
    X3d, y2d = load_train_tensors(args.data_type)
    print(f"  loaded train   : X{X3d.shape}, y{y2d.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    rng = np.random.default_rng(0)
    X, y = flatten_dataset(X3d, y2d, feature_cols, subsample=args.subsample, rng=rng)
    print(f"  flat shape     : X{X.shape}, y{y.shape} (subsampled) in {time.time()-t0:.1f}s")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=args.poly_degree, include_bias=False)),
        ("ridge", RidgeCV(alphas=args.alphas, store_cv_results=False)),
    ])

    t0 = time.time()
    pipe.fit(X, y)
    fit_time = time.time() - t0
    chosen_alpha = pipe.named_steps["ridge"].alpha_
    print(f"  fit done       : {fit_time:.1f}s   chosen alpha = {chosen_alpha}")

    # Quick training MSE
    y_pred = pipe.predict(X)
    train_rmse_ns = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    print(f"  train RMSE     : {train_rmse_ns*1000:.4f} ps "
          f"(on subsampled {X.shape[0]} pts)")

    suffix = args.out_suffix if args.out_suffix else f"poly{args.poly_degree}"
    out_path = os.path.join(
        MODEL_DIR,
        f"ridge_tsmc_combined_{args.data_type}_{suffix}.joblib",
    )
    joblib.dump({
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "poly_degree": args.poly_degree,
        "chosen_alpha": chosen_alpha,
        "subsample": args.subsample,
        "train_rmse_ns": train_rmse_ns,
        "data_type": args.data_type,
    }, out_path)
    print(f"  saved          : {out_path}")


if __name__ == "__main__":
    main()
