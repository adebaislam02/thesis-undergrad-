# -*- coding: utf-8 -*-
"""
Compute pairwise McNemar's test on acceptance-classification for the
one-shot-unmerged condition (where Haiku 4.5 achieves its best performance,
0.860 accuracy). Produces a CSV + LaTeX table showing which model
differences are statistically significant.
"""
import pandas as pd
import numpy as np
from scipy.stats import binom

BASE = "/Users/adeba/thesis-undergrad-/"

# Ground truth
gt = pd.read_csv("/Users/adeba/Downloads/FINAL_DATASET_FIXED.csv")
gt = gt[gt["pr_diff"].notna() & (gt["pr_diff"].str.strip() != "")].reset_index(drop=True)
gt["accepted_true"] = gt["pr_merged"].map({True: "yes", False: "no"})
gt["gt_index"] = gt.index

# All models on the one-shot-unmerged (1S-U) condition — Haiku's best
FILES = {
    "GPT-4o-mini":       "batch_one_shot_results_for_300_gpt4o_unmerged.csv",
    "Claude Haiku 4.5":  "_HAIKU45_batch_one_shot_unmerged_300.csv",
    "DeepSeek R1":       "batch_one_shot_results_UNmerged_deepseek.csv",
    "Mistral 3.2-24B":   "mistral_batch_one_shot_results_unmerged_mistral.csv",
}
MODELS = list(FILES.keys())


def load_paired(fname):
    df = pd.read_csv(BASE + fname, engine="python", on_bad_lines="skip")
    df["index"] = pd.to_numeric(df["index"], errors="coerce").astype("Int64")
    df = df[df["index"].notna()].copy()
    df["index"] = df["index"].astype(int)
    df = df.drop_duplicates(subset=["index"])
    return df


def mcnemar_exact(y_true, pred_a, pred_b):
    a_ok = (pred_a == y_true).to_numpy()
    b_ok = (pred_b == y_true).to_numpy()
    b_only = int((a_ok & ~b_ok).sum())    # A right, B wrong
    c_only = int((~a_ok & b_ok).sum())    # A wrong, B right
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    p = 2 * binom.cdf(min(b_only, c_only), n, 0.5)
    return b_only, c_only, min(1.0, p)


# Build a merged dataframe with all 4 models' predictions on shared rows
merged = gt[["gt_index", "accepted_true"]].copy()
for model in MODELS:
    df = load_paired(FILES[model])
    merged = merged.merge(
        df[["index", "accepted"]].rename(columns={"accepted": f"pred_{model}"}),
        left_on="gt_index", right_on="index", how="inner",
    ).drop(columns=["index"])
merged = merged.dropna().reset_index(drop=True)
n_shared = len(merged)
print(f"Shared rows across all 4 models: {n_shared}")

# All 6 pairwise comparisons
rows = []
for i in range(len(MODELS)):
    for j in range(i + 1, len(MODELS)):
        a, b = MODELS[i], MODELS[j]
        acc_a = (merged[f"pred_{a}"] == merged["accepted_true"]).mean()
        acc_b = (merged[f"pred_{b}"] == merged["accepted_true"]).mean()
        b_only, c_only, p = mcnemar_exact(
            merged["accepted_true"], merged[f"pred_{a}"], merged[f"pred_{b}"],
        )
        rows.append({
            "Model A": a, "Acc A": round(acc_a, 3),
            "Model B": b, "Acc B": round(acc_b, 3),
            "Δ": round(acc_a - acc_b, 3),
            "A wins": b_only, "B wins": c_only,
            "p-value": round(p, 4),
            "Significant (p<0.05)": "Yes" if p < 0.05 else "No",
        })

df_out = pd.DataFrame(rows)
csv_path = BASE + "results_table_mcnemar_pairwise.csv"
df_out.to_csv(csv_path, index=False)
print(f"\nWrote: {csv_path}")
print(df_out.to_string(index=False))

# --- Also produce LaTeX ---
def escape(s):
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")

body_lines = []
for _, r in df_out.iterrows():
    p_txt = f"{r['p-value']:.4f}"
    sig = r["Significant (p<0.05)"]
    p_disp = f"\\textbf{{{p_txt}}}" if sig == "Yes" else p_txt
    row = [
        escape(r["Model A"]), f"{r['Acc A']:.3f}",
        escape(r["Model B"]), f"{r['Acc B']:.3f}",
        f"{r['Δ']:+.3f}",
        str(r["A wins"]), str(r["B wins"]),
        p_disp,
    ]
    body_lines.append(" & ".join(row) + r" \\")

tex = (
    r"\begin{table}[!htbp]" + "\n"
    r"\centering" + "\n"
    r"\caption{Pairwise McNemar's test on the one-shot-unmerged "
    r"condition ($N=" + str(n_shared) + r"$ PRs on which all four models "
    r"returned valid predictions). ``$A$ wins'' counts PRs where model $A$ "
    r"was correct but $B$ was not; ``$B$ wins'' vice versa. $p$-values in "
    r"\textbf{bold} indicate a statistically significant accuracy gap "
    r"between the two models at $\alpha = 0.05$.}" + "\n"
    r"\label{tab:mcnemar}" + "\n"
    r"\small" + "\n"
    r"\setlength{\tabcolsep}{4pt}" + "\n"
    r"\begin{tabular}{lclccccl}" + "\n"
    r"\toprule" + "\n"
    r"Model $A$ & Acc $A$ & Model $B$ & Acc $B$ & $\Delta$ & $A$ wins & $B$ wins & $p$ \\" + "\n"
    r"\midrule" + "\n"
    + "\n".join(body_lines) + "\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}" + "\n"
    r"\end{table}" + "\n"
)

tex_path = BASE + "latex/table_5_mcnemar_pairwise.tex"
with open(tex_path, "w") as f:
    f.write(tex)
print(f"\nWrote: {tex_path}")
