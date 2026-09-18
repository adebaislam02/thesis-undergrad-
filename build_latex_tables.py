# -*- coding: utf-8 -*-
"""
Convert the eval driver's result CSVs into ready-to-paste LaTeX `table`
environments for the e-Informatica manuscript.

Output: one .tex snippet per table into /Users/adeba/thesis-undergrad-/latex/
"""

import pandas as pd
import os
import re

BASE = "/Users/adeba/thesis-undergrad-"
OUT = os.path.join(BASE, "latex")
os.makedirs(OUT, exist_ok=True)

# Short condition labels used throughout paper
COND_SHORT = {
    "zero-shot": "0-S",
    "one-shot-merged": "1S-M",
    "one-shot-unmerged": "1S-U",
    "few-shot": "Few-S",
    "—": "—",
}

# Short model labels for narrow tables (paper uses these)
MODEL_SHORT = {
    "GPT-4o-mini": "GPT-4o-mini",
    "Claude Haiku 4.5": "Haiku 4.5",
    "DeepSeek R1": "DeepSeek R1",
    "Mistral 3.2-24B": "Mistral 3.2-24B",
}


def escape(s):
    """Escape LaTeX special chars in a cell."""
    if s is None: return "—"
    if isinstance(s, float):
        if pd.isna(s): return "—"
        return f"{s:.3f}".rstrip("0").rstrip(".") if s != int(s) else str(int(s))
    s = str(s)
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def write_tex(name, content):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(content)
    print(f"  wrote {p}")


# ============================================================
# Table 5.1: Acceptance Classification (with baseline row)
# ============================================================
df = pd.read_csv(os.path.join(BASE, "results_table_5_1_acceptance.csv"))
df["cond_short"] = df["condition"].map(COND_SHORT)

body = []
prev_model = None
for _, r in df.iterrows():
    model = r["model"]
    if "baseline" in model.lower():
        model_disp = r"\textit{Majority baseline}"
    else:
        model_disp = MODEL_SHORT.get(model, model)
    # Only show model name on first row of that group
    if model == prev_model:
        model_cell = ""
    else:
        model_cell = model_disp
    prev_model = model
    # Format accuracy with 95% CI: "0.85 [0.81, 0.89]"
    if pd.notna(r.get("Acc_CI_lo")) and pd.notna(r.get("Acc_CI_hi")):
        acc_cell = f"{r['Accuracy']:.2f} [{r['Acc_CI_lo']:.2f}, {r['Acc_CI_hi']:.2f}]"
    else:
        acc_cell = f"{r['Accuracy']:.2f}"
    row = [
        model_cell, r["cond_short"], acc_cell,
        f"{r['P(Yes)']:.2f}", f"{r['R(Yes)']:.2f}", f"{r['F1(Yes)']:.2f}",
        f"{r['P(No)']:.2f}",  f"{r['R(No)']:.2f}",  f"{r['F1(No)']:.2f}",
        str(int(r['N'])),
    ]
    body.append(" & ".join(escape(x) for x in row) + r" \\")

tex = r"""\begin{table*}[t]
\centering
\caption{Acceptance classification performance. Acc.\ = Accuracy with
bootstrap 95\% confidence interval (2000 resamples); P = Precision,
R = Recall; Yes = merged, No = unmerged.
0-S = zero-shot, 1S-M = one-shot (merged example), 1S-U = one-shot
(unmerged example), Few-S = few-shot. The majority baseline always
predicts \emph{merged}. Pairwise statistical comparisons between
models are reported in Table~\ref{tab:mcnemar}.}
\label{tab:acceptance}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccccccc}
\toprule
Model & Cond. & Acc.\ [95\% CI] & P(Yes) & R(Yes) & F1(Yes) & P(No) & R(No) & F1(No) & N \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
write_tex("table_5_1_acceptance.tex", tex)


# ============================================================
# Table 5.2: Multi-label metrics
# ============================================================
df = pd.read_csv(os.path.join(BASE, "results_table_5_2_multilabel.csv"))
df["cond_short"] = df["condition"].map(COND_SHORT)

body = []
prev_model = None
for _, r in df.iterrows():
    model = r["model"]
    model_disp = MODEL_SHORT.get(model, model)
    if model == prev_model:
        model_cell = ""
    else:
        model_cell = model_disp
    prev_model = model
    row = [
        model_cell, r["cond_short"],
        f"{r['Macro-Precision']:.3f}", f"{r['Macro-Recall']:.3f}", f"{r['Macro-F1']:.3f}",
        f"{r['Micro-Precision']:.3f}", f"{r['Micro-Recall']:.3f}", f"{r['Micro-F1']:.3f}",
        f"{r['Hamming-Loss']:.3f}",    f"{r['Flattened-Acc']:.3f}", f"{r['Exact-Match-Acc']:.3f}",
    ]
    body.append(" & ".join(escape(x) for x in row) + r" \\")

tex = r"""\begin{table*}[t]
\centering
\caption{Multi-label flag generation performance. Macro/Micro averages
of Precision, Recall, and F1 over the 24-flag taxonomy; Hamming Loss;
flattened per-instance accuracy; and exact-match accuracy over the full
flag set.}
\label{tab:multilabel}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccccccc}
\toprule
Model & Cond. & Ma-P & Ma-R & Ma-F1 & Mi-P & Mi-R & Mi-F1 & Hamming & Flat.\ Acc & Exact-Match \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
write_tex("table_5_2_multilabel.tex", tex)


# ============================================================
# Table 5.3: Extractive vs. Evaluative (RQ3 headline)
# ============================================================
df = pd.read_csv(os.path.join(BASE, "results_table_extractive_vs_evaluative.csv"))
df["cond_short"] = df["condition"].map(COND_SHORT)

body = []
prev_model = None
for _, r in df.iterrows():
    model = r["model"]
    model_disp = MODEL_SHORT.get(model, model)
    if model == prev_model: model_cell = ""
    else: model_cell = model_disp
    prev_model = model
    row = [
        model_cell, r["cond_short"],
        f"{r['extractive_avg_F1']:.3f}",
        f"{r['evaluative_avg_F1']:.3f}",
        f"{r['gap']:+.3f}",
    ]
    body.append(" & ".join(escape(x) for x in row) + r" \\")

tex = r"""\begin{table}[!htbp]
\centering
\caption{Average per-flag F1 within the two flag categories defined in
Section~\ref{sec:Method}: \emph{extractive} (4 flags with explicit
numeric thresholds in the prompt) versus \emph{evaluative} (20 flags
requiring semantic judgment). The gap column reports
(extractive $-$ evaluative).}
\label{tab:extractive-evaluative}
\small
\begin{tabular}{llccc}
\toprule
Model & Cond. & Extractive & Evaluative & Gap \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
write_tex("table_5_3_extractive_vs_evaluative.tex", tex)


# ============================================================
# Table 5.4: Vocabulary drift
# ============================================================
df = pd.read_csv(os.path.join(BASE, "results_table_vocab_drift.csv"))
df["cond_short"] = df["condition"].map(COND_SHORT)

body = []
prev_model = None
for _, r in df.iterrows():
    model = r["model"]
    model_disp = MODEL_SHORT.get(model, model)
    if model == prev_model: model_cell = ""
    else: model_cell = model_disp
    prev_model = model
    strict_f1 = f"{r['strict_macro_F1']:.3f}"
    norm_f1 = f"{r['normalized_macro_F1']:.3f}" if pd.notna(r['normalized_macro_F1']) else "—"
    row = [
        model_cell, r["cond_short"],
        f"{r['rows_with_non_vocab_flag']}",
        f"{r['pct_rows_with_non_vocab_flag']:.1f}%",
        f"{r['total_non_vocab_emissions']}",
        strict_f1,
        norm_f1,
    ]
    body.append(" & ".join(escape(x) for x in row) + r" \\")

tex = r"""\begin{table*}[t]
\centering
\caption{Vocabulary drift by model and prompting condition. \emph{Rows}
counts predictions containing at least one flag name outside the enumerated
vocabulary. \emph{Emissions} counts the total non-vocabulary flag tokens
across all predictions. \emph{Strict F1} is the macro-F1 from
Table~\ref{tab:multilabel}; \emph{Norm.\ F1} is the same metric after
mapping case- and separator-variants to their vocabulary form
(e.g., ``Poor code quality''~$\rightarrow$~\texttt{poor\_code\_quality}).}
\label{tab:vocab-drift}
\small
\begin{tabular}{llrrrrr}
\toprule
Model & Cond. & Drift rows & \% & Emissions & Strict F1 & Norm.\ F1 \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
write_tex("table_5_4_vocab_drift.tex", tex)


# ============================================================
# Per-model per-flag F1 tables (5.5–5.8)
# ============================================================
for model, fname_slug, label_num in [
    ("GPT-4o-mini",       "gpt_4o_mini",       5),
    ("Claude Haiku 4.5",  "claude_haiku_4_5",  6),
    ("DeepSeek R1",       "deepseek_r1",       7),
    ("Mistral 3.2-24B",   "mistral_3_2_24b",   8),
]:
    src = os.path.join(BASE, f"results_table_5_perflag_{fname_slug}.csv")
    if not os.path.exists(src): continue
    df = pd.read_csv(src)
    # Reorder columns
    cond_cols = ["zero-shot", "one-shot-merged", "one-shot-unmerged", "few-shot"]
    body = []
    for _, r in df.iterrows():
        flag = r["flag"]
        # Wrap flag name in \texttt{} and escape underscores
        flag_tex = r"\texttt{" + flag.replace("_", r"\_") + r"}"
        vals = []
        for c in cond_cols:
            v = r[c]
            vals.append(f"{v:.2f}" if pd.notna(v) else "—")
        avg = r.get("Average", None)
        avg_str = f"{avg:.2f}" if pd.notna(avg) else "—"
        body.append(f"{flag_tex} & " + " & ".join(vals) + f" & {avg_str} " + r"\\")

    tex = r"""\begin{table}[!htbp]
\centering
\caption{Per-flag F1 for """ + model + r""" across the four prompting
conditions. Rows are the 24 flags in the taxonomy defined in
Section~\ref{sec:Method}; the \emph{Average} column is the mean across
the four conditions.}
\label{tab:perflag-""" + fname_slug.replace("_", "-") + r"""}
\small
\begin{tabular}{lccccc}
\toprule
Flag & 0-S & 1S-M & 1S-U & Few-S & Avg. \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_tex(f"table_5_{label_num}_perflag_{fname_slug}.tex", tex)


# ============================================================
# Master file that \input{}s all of them (optional convenience)
# ============================================================
master = r"""% Include this from your main manuscript, e.g. in Section 5.
% Adjust file paths if you don't keep the .tex files alongside main.tex.
\input{table_5_1_acceptance.tex}
\input{table_5_2_multilabel.tex}
\input{table_5_3_extractive_vs_evaluative.tex}
\input{table_5_4_vocab_drift.tex}
\input{table_5_5_perflag_gpt_4o_mini.tex}
\input{table_5_6_perflag_claude_haiku_4_5.tex}
\input{table_5_7_perflag_deepseek_r1.tex}
\input{table_5_8_perflag_mistral_3_2_24b.tex}
"""
write_tex("_ALL_TABLES.tex", master)

print("\nDone. All .tex files in /Users/adeba/thesis-undergrad-/latex/")
