import pandas as pd
import ast
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, hamming_loss

# Load the ground truth and predictions
gt = pd.read_csv("FINAL_DATASET - 300_Annotated.csv")  # Replace with your ground truth file path
pred = pd.read_csv("batch_one_shot_results_for_300_gpt4o.csv")  # Replace with your predictions file path

# --- 2. Label mappings ---
H2I_RED = {
    "PR not aligned with issue":             "pr_not_aligned_with_issue",
    "Missing or inadequate test cases":      "missing_or_inadequate_test_cases",
    "Low contributor acceptance rate":       "low_contributor_acceptance_rate",
    "Negative sentiment in comments":        "negative_sentiment_in_comments",
    "Large number of changed lines":         "large_number_of_changed_lines",
    "Unclear or missing PR title":           "unclear_or_missing_pr_title",
    "Unclear or missing PR description":     "unclear_or_missing_pr_description",
    "Unclear or missing commit messages":    "unclear_or_missing_commit_messages",
    "Poor code quality":                      "poor_code_quality",
    "Unresponsive after feedback":            "unresponsive_after_feedback",
    "Fails tests":                           "fails_tests",
    "Risks breaking compatibility":          "breaks_compatibility",
    "Solution is incorrect or inefficient":  "solution_is_incorrect_or_inefficient",
    "Overengineered solution":               "overengineered_solution",
}
H2I_GREEN = {
    "Clear alignment with issue":            "clear_alignment_with_issue",
    "Includes test cases":                   "includes_test_cases",
    "High contributor acceptance rate":      "high_contributor_acceptance_rate",
    "Positive sentiment in comments":        "positive_sentiment_in_comments",
    "Manageable number of changed lines":    "manageable_number_of_changed_lines",
    "Clear PR title":                        "clear_pr_title",
    "Clear PR description":                  "clear_pr_description",
    "Clear commit message":                  "clear_commit_message",
    "Good code quality":                     "good_code_quality",
    "Responsive to feedback":                "responsive_to_feedback",
}

# --- 3. Parsing logic for human labels ---
def parse_human_flags(cell, mapping):
    if pd.isna(cell) or not str(cell).strip():
        return []
    items = [s.strip() for s in str(cell).split(",") if s.strip()]
    return [mapping[s] for s in items if s in mapping]

# --- 4. Final/fallback selection for labels ---
def pick_final_or_annotator_flags(row, final_col, annot2_col, annot1_col, mapping):
    final_val = row.get(final_col, "")
    if isinstance(final_val, str) and final_val.strip():
        return parse_human_flags(final_val, mapping)
    annot2_val = row.get(annot2_col, "")
    if isinstance(annot2_val, str) and annot2_val.strip():
        return parse_human_flags(annot2_val, mapping)
    annot1_val = row.get(annot1_col, "")
    if isinstance(annot1_val, str) and annot1_val.strip():
        return parse_human_flags(annot1_val, mapping)
    return []

gt["red_flags_true"] = gt.apply(
    lambda row: pick_final_or_annotator_flags(
        row,
        "Red Flags (Final)",
        "Red Flags (Annotator 2)",
        "Red Flags (Annotator 1)",
        H2I_RED,
    ),
    axis=1,
)
gt["green_flags_true"] = gt.apply(
    lambda row: pick_final_or_annotator_flags(
        row,
        "Green Flags (Final)",
        "Green Flags (Annotator 2)",
        "Green Flags (Annotator 1)",
        H2I_GREEN,
    ),
    axis=1,
)

gt["accepted_true"] = gt["pr_merged"].map({True: "yes", False: "no"})

# --- 5. Load predictions ---
pred = pd.read_csv("batch_one_shot_results_for_300_gpt4o.csv")  # Replace with your predictions file path
pred = pred.dropna(subset=["accepted"]).reset_index(drop=True)

def to_list(cell):
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except Exception:
            return []
    return []

pred["red_flags_pred"] = pred["red_flags"].apply(to_list)
pred["green_flags_pred"] = pred["green_flags"].apply(to_list)
pred["accepted_pred"] = pred["accepted"].astype(str)

# --- 6. Merge predictions and ground-truth on indices ---
df = gt.merge(
    pred,
    left_on="index_true",
    right_on="index",
    how="inner",
    suffixes=("_gt", "_pred"),
).reset_index(drop=True)

# Sanity ensure predicted flag columns are lists
df["red_flags_pred"] = df["red_flags_pred"].apply(lambda x: x if isinstance(x, list) else [])
df["green_flags_pred"] = df["green_flags_pred"].apply(lambda x: x if isinstance(x, list) else [])

# --- 7a. Acceptance classification report ---
print("=== ACCEPTANCE CLASSIFICATION ===")
print(
    classification_report(
        df["accepted_true"],
        df["accepted_pred"],
        labels=["yes", "no"],
        zero_division=0,
    )
)
print("Accuracy:", accuracy_score(df["accepted_true"], df["accepted_pred"]))

# --- 7b. Per-flag precision, recall, F1 ---
ALL_FLAGS = list(H2I_RED.values()) + list(H2I_GREEN.values())
rows = []
for flag in ALL_FLAGS:
    if flag in H2I_RED.values():
        y_true = df["red_flags_true"].apply(lambda L: flag in L)
        y_pred = df["red_flags_pred"].apply(lambda L: flag in L)
    else:
        y_true = df["green_flags_true"].apply(lambda L: flag in L)
        y_pred = df["green_flags_pred"].apply(lambda L: flag in L)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    rows.append(
        {"flag": flag, "precision": p, "recall": r, "f1": f1, "support": int(y_true.sum())}
    )

flags_df = pd.DataFrame(rows)
print("\n=== PER-FLAG METRICS ===")
print(flags_df)
print("\nMacro-avg F1:", flags_df["f1"].mean())

# --- 7c. Multi-label combined metrics ---
y_true = []
y_pred = []
for _, row in df.iterrows():
    true_flags = set(row["red_flags_true"] + row["green_flags_true"])
    pred_flags = set(row["red_flags_pred"] + row["green_flags_pred"])
    y_true.append([1 if f in true_flags else 0 for f in ALL_FLAGS])
    y_pred.append([1 if f in pred_flags else 0 for f in ALL_FLAGS])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="micro", zero_division=0
)
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
hloss = hamming_loss(y_true, y_pred)

print("\n=== OVERALL MULTI-LABEL METRICS ===")
print(f"Micro P/R/F1: {p_micro:.3f} / {r_micro:.3f} / {f1_micro:.3f}")
print(f"Macro P/R/F1: {p_macro:.3f} / {r_macro:.3f} / {f1_macro:.3f}")
print(f"Hamming Loss: {hloss:.3f}")

# --- 7d. Flattened accuracy and exact-match accuracy ---
flat_acc = (y_true == y_pred).mean()
subset_acc = np.all(y_true == y_pred, axis=1).mean()

print(f"Flattened flag accuracy (TP+TN over all label instances): {flat_acc:.3f}")
print(f"Exact-match flag accuracy (all flags correct per PR):    {subset_acc:.3f}")






