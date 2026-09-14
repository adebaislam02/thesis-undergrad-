# -*- coding: utf-8 -*-
"""
Compute per-flag Cohen's Kappa between Annotator 1 and Annotator 2,
extending your existing Validation notebook logic. Produces:
 - `results_table_perflag_kappa.csv`  (24-row table)
 - `latex/table_perflag_kappa.tex`    (paste-ready LaTeX)

Only PRs where BOTH annotators labeled the flag are used.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DATASET = "/Users/adeba/Downloads/FINAL_DATASET_FIXED.csv"

# The 24 flags in canonical form → human-readable string used by annotators
FLAG_MAP = {
    "pr_not_aligned_with_issue":            "PR not aligned with issue",
    "missing_or_inadequate_test_cases":     "Missing or inadequate test cases",
    "low_contributor_acceptance_rate":      "Low contributor acceptance rate",
    "negative_sentiment_in_comments":       "Negative sentiment in comments",
    "large_number_of_changed_lines":        "Large number of changed lines",
    "unclear_or_missing_pr_title":          "Unclear or missing PR title",
    "unclear_or_missing_pr_description":    "Unclear or missing PR description",
    "unclear_or_missing_commit_messages":   "Unclear or missing commit messages",
    "poor_code_quality":                    "Poor code quality",
    "unresponsive_after_feedback":          "Unresponsive after feedback",
    "fails_tests":                          "Fails tests",
    "breaks_compatibility":                 "Risks breaking compatibility",
    "solution_incorrect_or_inefficient":    "Solution is incorrect or inefficient",
    "overengineered_solution":              "Overengineered solution",
    "clear_alignment_with_issue":           "Clear alignment with issue",
    "includes_test_cases":                  "Includes test cases",
    "high_contributor_acceptance_rate":     "High contributor acceptance rate",
    "positive_sentiment_in_comments":       "Positive sentiment in comments",
    "manageable_number_of_changed_lines":   "Manageable number of changed lines",
    "clear_pr_title":                       "Clear PR title",
    "clear_pr_description":                 "Clear PR description",
    "clear_commit_message":                 "Clear commit message",
    "good_code_quality":                    "Good code quality",
    "responsive_to_feedback":               "Responsive to feedback",
}
RED_FLAGS = set(list(FLAG_MAP.keys())[:14])
GREEN_FLAGS = set(list(FLAG_MAP.keys())[14:])


def parse_flags(cell):
    """Split a comma-separated annotator string into a set of trimmed labels."""
    if pd.isna(cell) or not str(cell).strip():
        return set()
    return {s.strip() for s in str(cell).split(",") if s.strip()}


# --- Load data ---
df = pd.read_csv(DATASET)
# Only use PRs where BOTH annotators labeled the flags (i.e., the 20% double-annotated subset)
df = df.dropna(subset=[
    "Red Flags (Annotator 1)", "Red Flags (Annotator 2)",
    "Green Flags (Annotator 1)", "Green Flags (Annotator 2)",
], how="all")
# Actually be stricter — require A2 to be non-empty (indicates it was double-annotated)
mask = (
    df["Red Flags (Annotator 2)"].fillna("").astype(str).str.strip().ne("") |
    df["Green Flags (Annotator 2)"].fillna("").astype(str).str.strip().ne("")
)
df = df[mask].reset_index(drop=True)
print(f"Double-annotated PRs (rows with any A2 flag): {len(df)}")

# --- For each flag, compute binary yes/no per PR from each annotator, then kappa ---
rows = []
for internal_flag, human_label in FLAG_MAP.items():
    is_red = internal_flag in RED_FLAGS
    if is_red:
        a1_col = "Red Flags (Annotator 1)"
        a2_col = "Red Flags (Annotator 2)"
    else:
        a1_col = "Green Flags (Annotator 1)"
        a2_col = "Green Flags (Annotator 2)"

    a1_yes = df[a1_col].apply(lambda c: 1 if human_label in parse_flags(c) else 0)
    a2_yes = df[a2_col].apply(lambda c: 1 if human_label in parse_flags(c) else 0)

    # Cohen's Kappa on binary labels
    # If all values are same (e.g., all zero), kappa is undefined → sklearn returns 0 or NaN
    if len(set(a1_yes) | set(a2_yes)) < 2:
        kappa = float("nan")
    else:
        kappa = cohen_kappa_score(a1_yes, a2_yes)

    both_positive = int((a1_yes & a2_yes).sum())
    a1_only = int(((a1_yes == 1) & (a2_yes == 0)).sum())
    a2_only = int(((a1_yes == 0) & (a2_yes == 1)).sum())
    n = len(df)

    rows.append({
        "flag": internal_flag,
        "polarity": "red" if is_red else "green",
        "kappa": round(kappa, 3) if pd.notna(kappa) else None,
        "N_double_annotated": n,
        "both_positive": both_positive,
        "A1_only_positive": a1_only,
        "A2_only_positive": a2_only,
        "interpretation": (
            "almost perfect" if kappa >= 0.81 else
            "substantial"    if kappa >= 0.61 else
            "moderate"       if kappa >= 0.41 else
            "fair"           if kappa >= 0.21 else
            "slight"         if kappa >= 0.00 else
            "poor"
        ) if pd.notna(kappa) else "undefined",
    })

result = pd.DataFrame(rows)
result_path = "/Users/adeba/thesis-undergrad-/results_table_perflag_kappa.csv"
result.to_csv(result_path, index=False)
print(f"\nWrote: {result_path}")
print(result.to_string(index=False))

# Summary
print("\n=== SUMMARY ===")
print(f"Mean kappa (all 24 flags):        {result['kappa'].mean():.3f}")
print(f"Mean kappa (14 red flags):        {result[result['polarity']=='red']['kappa'].mean():.3f}")
print(f"Mean kappa (10 green flags):      {result[result['polarity']=='green']['kappa'].mean():.3f}")
print(f"Flags with 'almost perfect' agreement (κ ≥ 0.81): {(result['kappa'] >= 0.81).sum()}")
print(f"Flags with 'substantial'      agreement (κ ≥ 0.61): {(result['kappa'] >= 0.61).sum()}")
print(f"Flags with 'moderate'         agreement (κ ≥ 0.41): {(result['kappa'] >= 0.41).sum()}")
print(f"Flags with 'fair'             agreement (κ < 0.41): {(result['kappa'] < 0.41).sum()}")

# --- LaTeX output ---
def escape(s):
    return str(s).replace("_", r"\_")

body_lines = []
# Sort: red flags first (in vocab order), then green (in vocab order)
for polarity in ["red", "green"]:
    sub = result[result["polarity"] == polarity]
    for _, r in sub.iterrows():
        kappa_str = f"{r['kappa']:.3f}" if pd.notna(r["kappa"]) else "---"
        row = [
            r"\texttt{" + escape(r["flag"]) + r"}",
            r["polarity"],
            kappa_str,
            str(r["both_positive"]),
            str(r["A1_only_positive"]),
            str(r["A2_only_positive"]),
            r["interpretation"],
        ]
        body_lines.append(" & ".join(row) + r" \\")
    if polarity == "red":
        body_lines.append(r"\midrule")

n_da = len(df)
tex = (
    r"\begin{table*}[!htbp]" + "\n"
    r"\centering" + "\n"
    r"\caption{Per-flag Cohen's Kappa between Annotator 1 and Annotator 2 on "
    r"the double-annotated subset ($N=" + str(n_da) + r"$ PRs). "
    r"\emph{Both~+} counts PRs where both annotators applied the flag; "
    r"\emph{A1~only} and \emph{A2~only} count PRs where exactly one annotator did. "
    r"Interpretation follows Landis \& Koch (1977): $\kappa \geq 0.81$ "
    r"almost perfect, $0.61$--$0.80$ substantial, $0.41$--$0.60$ moderate, "
    r"$0.21$--$0.40$ fair.}" + "\n"
    r"\label{tab:perflag-kappa}" + "\n"
    r"\small" + "\n"
    r"\setlength{\tabcolsep}{5pt}" + "\n"
    r"\begin{tabular}{llccccl}" + "\n"
    r"\toprule" + "\n"
    r"Flag & Polarity & $\kappa$ & Both+ & A1 only & A2 only & Interpretation \\" + "\n"
    r"\midrule" + "\n"
    + "\n".join(body_lines) + "\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}" + "\n"
    r"\end{table*}" + "\n"
)

tex_path = "/Users/adeba/thesis-undergrad-/latex/table_perflag_kappa.tex"
with open(tex_path, "w") as f:
    f.write(tex)
print(f"\nWrote: {tex_path}")
