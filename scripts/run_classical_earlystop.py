"""Huấn luyện MLP và MLP_tiny với CƠ CHẾ EARLY STOPPING GIỐNG QNN.

Mục đích: tạo baseline cổ điển dùng đúng quy trình dừng sớm như QNN, để so sánh
công bằng với các bảng kết quả VQC.

Cơ chế (sao y src/training/qnn_trainer.py, phần early stopping):
    - tách stratified validation 0.2 từ X_train (train_test_split, random_state=seed)
      -> CÙNG các mẫu validation với QNN ở mỗi seed.
    - huấn luyện trên 0.8 còn lại bằng Adam, mỗi bước = 1 epoch (partial_fit).
    - mỗi epoch tính validation BCE (= log_loss); GIỮ trọng số tốt nhất theo val BCE.
    - dừng nếu val BCE không cải thiện > min_delta sau `patience` epoch; KHÔI PHỤC
      trọng số tốt nhất trước khi trả về. Test set không bị đụng tới.

Lưu ý: theo dõi val LOSS (không phải accuracy). sklearn MLPClassifier(early_stopping
=True) mặc định dừng theo validation ACCURACY -> dừng quá sớm trên dữ liệu cân bằng,
nên ta tự viết vòng lặp này để khớp đúng tiêu chí của QNN.

Kiến trúc (giống baseline trong src/models/classical.py, chỉ thêm early stopping):
    mlp       -> hidden (32, 16)
    mlp_tiny  -> hidden (4,)   # cùng cỡ tham số với VQC (~25 params)

Chạy:  python scripts/run_classical_earlystop.py
Đầu ra: tables/mlp_es_val_n500.csv  (mean 5 seed cho S1 & S3, N=500)
"""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data.scenarios import build_scenario
from src.evaluation.metrics import compute_train_test_metrics

# ---- Tham số early stopping (khớp QNN) -------------------------------------
VAL_FRACTION = 0.2
PATIENCE = 15
MIN_DELTA = 0.0
MAX_EPOCHS = 500

ARCHES = {"mlp": (32, 16), "mlp_tiny": (4,)}
METRICS = [
    "test_accuracy", "test_f1", "test_precision_attack", "test_recall_attack",
    "test_recall_benign", "test_roc_auc", "test_fpr",
    "train_accuracy", "train_f1", "gap_f1", "gap_accuracy",
]


def _n_params(model) -> int:
    return sum(int(np.prod(w.shape)) for w in model.coefs_) + \
           sum(int(np.prod(b.shape)) for b in model.intercepts_)


def train_like_qnn(X_train, y_train, seed, hidden):
    """MLP + early stopping trên validation BCE, khôi phục trọng số tốt nhất."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_FRACTION,
        stratify=y_train, random_state=seed,
    )
    model = MLPClassifier(
        hidden_layer_sizes=hidden, solver="adam",
        random_state=seed, warm_start=True, max_iter=1,
    )
    best_val, best_weights, best_epoch, no_improve = np.inf, None, None, 0
    for epoch in range(MAX_EPOCHS):
        model.partial_fit(X_fit, y_fit, classes=[0, 1])  # 1 epoch Adam
        val_loss = log_loss(y_val, model.predict_proba(X_val)[:, 1], labels=[0, 1])
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_weights = (copy.deepcopy(model.coefs_),
                            copy.deepcopy(model.intercepts_))
            best_epoch, no_improve = epoch, 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    if best_weights is not None:  # khôi phục trọng số tốt nhất theo val BCE
        model.coefs_, model.intercepts_ = best_weights
    return model, best_val, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500,
                        help="train_samples_per_class (mặc định 500)")
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", default=None,
                        help="đường dẫn CSV (mặc định tables/mlp_es_val_n<N>.csv)")
    args = parser.parse_args()

    rows = []
    for name, hidden in ARCHES.items():
        for scn in args.scenarios:
            for seed in args.seeds:
                data = build_scenario(
                    scenario=scn, train_per_class=args.n,
                    test_per_class=args.n, n_features=args.n_features, seed=seed,
                )
                model, best_val, best_epoch = train_like_qnn(
                    data["X_train"], data["y_train"], seed, hidden)
                y_tr_pred = model.predict(data["X_train"])
                y_te_pred = model.predict(data["X_test"])
                y_tr_prob = model.predict_proba(data["X_train"])[:, 1]
                y_te_prob = model.predict_proba(data["X_test"])[:, 1]
                m = compute_train_test_metrics(
                    data["y_train"], y_tr_pred, y_tr_prob,
                    data["y_test"], y_te_pred, y_te_prob)
                rows.append({"model": name, "scenario": scn, "seed": seed,
                             "best_epoch": best_epoch, "n_params": _n_params(model),
                             **{k: m[k] for k in METRICS}})
                print(f"{name:9s} {scn} seed{seed}: "
                      f"test_f1={m['test_f1']:.4f} best_epoch={best_epoch}")

    raw = pd.DataFrame(rows)

    # Tổng hợp: mean 5 seed + std cho 2 metric chính.
    agg = []
    for name in ARCHES:
        for scn in args.scenarios:
            g = raw[(raw["model"] == name) & (raw["scenario"] == scn)]
            row = {"model": name, "scenario": scn, "n_seeds": len(g),
                   "n_params": int(g["n_params"].iloc[0]),
                   "best_epoch_mean": round(g["best_epoch"].mean(), 1)}
            for k in METRICS:
                row[k] = round(g[k].mean(), 4)
            row["test_acc_std"] = round(g["test_accuracy"].std(ddof=1), 4)
            row["test_f1_std"] = round(g["test_f1"].std(ddof=1), 4)
            agg.append(row)

    cols = ["model", "scenario", "n_seeds", "n_params", "best_epoch_mean",
            "test_accuracy", "test_acc_std", "test_f1", "test_f1_std",
            "test_precision_attack", "test_recall_attack", "test_recall_benign",
            "test_roc_auc", "test_fpr", "train_accuracy", "train_f1",
            "gap_f1", "gap_accuracy"]
    out_df = pd.DataFrame(agg)[cols]

    out = Path(args.output) if args.output else REPO / "tables" / f"mlp_es_val_n{args.n}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print("\n=== MLP & MLP_tiny, early stopping kiểu QNN (val BCE), 5 seed ===")
    print(out_df.to_string(index=False))
    print(f"\nĐã ghi: {out}")


if __name__ == "__main__":
    main()
