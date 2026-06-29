"""Sinh lại TẤT CẢ bảng kết quả dùng trong chương Thực nghiệm, ghi vào tables/.

Mục đích: kiểm chứng (audit) các con số trong LaTeX. Mỗi bảng trong chương
(Kết quả 1/2/3) được tái tạo từ results_summary.csv với logic chọn cấu hình
trình bày tường minh bên dưới; xuất ra CSV (mean và std tách riêng, làm tròn 3
chữ số) trùng bố cục bảng tương ứng để so trực tiếp với file .tex.

Chạy:  python scripts/gen_result_tables.py
Đầu ra (folder tables/):
    kq1_base_q1-q4_n500_S1.csv         <- Bảng "Kết quả 1" (tab:kq1)
    kq2_vqc_vs_classical_n500.csv      <- Bảng "Kết quả 2" (tab:kq2)
    kq3_ablation_q2.csv                <- Bảng ablation q2 (tab:kq3-q2)
    kq3_ablation_q3.csv                <- Bảng ablation q3 (tab:kq3-q3)
    kq3_ablation_q4.csv                <- Bảng ablation q4 (tab:kq3-q4)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results_summary.csv"
OUT = REPO / "tables"
ROUND = 3

# ---- Định nghĩa kiến trúc QNN (encoding, ansatz) ---------------------------
QNN_ARCH = {
    "q1": ("angle", "rotation_only"),        # không entanglement
    "q2": ("angle", "basic_entangler"),
    "q3": ("angle", "strongly_entangling"),
    "q4": ("reuploading", "basic_entangler"),
}
# ---- Biến thể readout/encoding (ablation) ----------------------------------
#   z     : readout == 'z' và KHÔNG trainable encoding  (cơ sở)
#   z+zz  : readout == 'z+zz'
#   probs : readout == 'probs'
#   tenc  : trainable_encoding == True (readout z)
VARIANTS = ["z", "z+zz", "probs", "tenc"]


def _load() -> pd.DataFrame:
    if not SRC.exists():
        raise SystemExit(f"Không thấy {SRC}. Chạy scripts/aggregate.py trước.")
    return pd.read_csv(SRC)


def _bool(series):
    # trainable_encoding có thể lưu dạng True/False hoặc 'True'/'False'
    return series.astype(str).str.lower().isin(["true", "1"])


def sel_qnn(df, arch, variant, scenario, N):
    """Lọc các run QNN cho (kiến trúc, biến thể, kịch bản, N)."""
    enc, ans = QNN_ARCH[arch]
    m = (
        (df.model_type == "qnn")
        & (df.encoding == enc)
        & (df.ansatz == ans)
        & (df.scenario == scenario)
        & (df.train_samples_per_class == N)
    )
    te = _bool(df.trainable_encoding)
    if variant == "z":
        m &= (df.readout == "z") & (~te)
    elif variant == "z+zz":
        m &= (df.readout == "z+zz")
    elif variant == "probs":
        m &= (df.readout == "probs")
    elif variant == "tenc":
        m &= te
    else:
        raise ValueError(variant)
    return df[m]


def sel_classical(df, name, scenario, N):
    return df[
        (df.model_type == "classical")
        & (df.model_name == name)
        & (df.scenario == scenario)
        & (df.train_samples_per_class == N)
    ]


def stats(sub):
    """Trả về dict mean/std (3 dp) cho F1, FNR, FPR + n_seeds, n_params."""
    if len(sub) == 0:
        return None
    g = lambda col: (round(sub[col].mean(), ROUND), round(sub[col].std(ddof=1), ROUND))
    f1m, f1s = g("test_f1")
    fnm, fns = g("test_fnr")
    fpm, fps = g("test_fpr")
    return {
        "n_seeds": len(sub),
        "n_params": int(round(sub["n_params"].mean())),
        "F1_mean": f1m, "F1_std": f1s,
        "FNR_mean": fnm, "FNR_std": fns,
        "FPR_mean": fpm, "FPR_std": fps,
    }


# ===========================================================================
#  Bảng Kết quả 1 (tab:kq1): q1-q4 cơ sở (readout z), S1, N=500
# ===========================================================================
def table_kq1(df):
    rows = []
    for q in ["q1", "q2", "q3", "q4"]:
        s = stats(sel_qnn(df, q, "z", "S1", 500))
        if s is None:
            print(f"  [kq1] CẢNH BÁO: thiếu dữ liệu cho {q}")
            continue
        rows.append({"model": q, **s})
    return pd.DataFrame(rows)


# ===========================================================================
#  Bảng Kết quả 2 (tab:kq2): VQC tốt nhất vs classical, N=500, S1 & S3
# ===========================================================================
def table_kq2(df):
    # (nhãn hiển thị, hàm lọc)
    specs = [
        ("Random Forest", lambda sc: sel_classical(df, "rf", sc, 500)),
        ("MLP (32,16)",   lambda sc: sel_classical(df, "mlp", sc, 500)),
        ("MLP-tiny (4)",  lambda sc: sel_classical(df, "mlp_tiny", sc, 500)),
        ("q4 <Z>",        lambda sc: sel_qnn(df, "q4", "z", sc, 500)),
        ("q4 <Z>+<ZZ>",   lambda sc: sel_qnn(df, "q4", "z+zz", sc, 500)),
    ]
    rows = []
    for label, fn in specs:
        rec = {"model": label}
        for sc in ["S1", "S3"]:
            s = stats(fn(sc))
            if s is None:
                print(f"  [kq2] CẢNH BÁO: thiếu dữ liệu {label} / {sc}")
                continue
            # n_params báo cáo RIÊNG mỗi kịch bản: với Random Forest số node phụ
            # thuộc dữ liệu huấn luyện nên S1 và S3 KHÁC nhau; các mô hình tham số
            # (MLP, VQC) thì giống nhau.
            rec[f"{sc}_nparams"] = s["n_params"]
            for k in ["F1_mean", "F1_std", "FNR_mean", "FNR_std", "FPR_mean", "FPR_std"]:
                rec[f"{sc}_{k}"] = s[k]
        rows.append(rec)
    cols = ["model"] + [
        f"{sc}_{k}" for sc in ["S1", "S3"]
        for k in ["nparams", "F1_mean", "F1_std", "FNR_mean", "FNR_std", "FPR_mean", "FPR_std"]
    ]
    return pd.DataFrame(rows)[cols]


# ===========================================================================
#  Bảng Kết quả 3 (tab:kq3-q2/q3/q4): ablation theo N trên S1 & S3 (F1)
# ===========================================================================
def table_kq3(df, arch):
    rows = []
    for v in VARIANTS:
        rec = {"variant": v}
        for sc in ["S1", "S3"]:
            for N in [100, 250, 500]:
                s = stats(sel_qnn(df, arch, v, sc, N))
                if s is None:
                    rec[f"{sc}_N{N}_F1_mean"] = ""
                    rec[f"{sc}_N{N}_F1_std"] = ""
                else:
                    rec[f"{sc}_N{N}_F1_mean"] = s["F1_mean"]
                    rec[f"{sc}_N{N}_F1_std"] = s["F1_std"]
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    df = _load()
    OUT.mkdir(parents=True, exist_ok=True)

    outputs = {
        "kq1_base_q1-q4_n500_S1.csv": table_kq1(df),
        "kq2_vqc_vs_classical_n500.csv": table_kq2(df),
        "kq3_ablation_q2.csv": table_kq3(df, "q2"),
        "kq3_ablation_q3.csv": table_kq3(df, "q3"),
        "kq3_ablation_q4.csv": table_kq3(df, "q4"),
    }
    for fname, tab in outputs.items():
        path = OUT / fname
        tab.to_csv(path, index=False)
        print(f"\n=== {fname}  ->  {path} ===")
        print(tab.to_string(index=False))

    print(f"\nĐã ghi {len(outputs)} bảng vào {OUT}")


if __name__ == "__main__":
    main()
