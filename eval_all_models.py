# -*- coding: utf-8 -*-
"""
Generalized evaluation driver for the 4-model LLM PR-review comparison.

Models: GPT-4o-mini, Claude Haiku 4.5, DeepSeek R1, Mistral Small 3.2-24B
Conditions: zero-shot, one-shot-merged, one-shot-unmerged, few-shot

Ground truth:  FINAL_DATASET_FIXED.csv with Final → Annotator 2 → Annotator 1
               fallback chain.

Special cases:
  * DeepSeek zero-shot — CSV is corrupted (multi-line per record with broken
    quoting). Metrics loaded directly from paper's saved .txt eval output.
  * Mistral — computes BOTH strict and normalized F1. Normalization handles
    case + space→underscore + a few known variants. Delta between strict &
    normalized reported as a "vocabulary drift" finding.
  * `solution_is_incorrect_or_inefficient` (paper's GT normalization with
    `is_`) is renamed to `solution_incorrect_or_inefficient` (model vocab)
    to fix a naming bug that forced 0.00 F1 on this flag in the paper.
  * Failed predictions (NaN accepted) are excluded per paper Ch.4 policy;
    the effective N per (model, condition) is reported.

Outputs (CSVs in project dir):
  results_table_5_1_acceptance.csv
  results_table_5_2_multilabel.csv
  results_table_5_3_perflag_all_models.csv
  results_table_vocab_drift.csv
  results_exclusion_counts.csv
"""

import pandas as pd
import numpy as np
import ast
import os
import re
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, hamming_loss,
    classification_report,
)

# ============================================================
# Bootstrap 95% CI on accuracy
# ============================================================
def bootstrap_accuracy_ci(y_true, y_pred, n_boot=2000, seed=42):
    """Bootstrap 95% CI on paired accuracy."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    accs = (y_true[idx] == y_pred[idx]).mean(axis=1)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)

BASE = "/Users/adeba/thesis-undergrad-"
GT_PATH = "/Users/adeba/Downloads/FINAL_DATASET_FIXED.csv"
DS_ZS_TXT = "/Users/adeba/Downloads/eval/deepseek_batch_zero_shot_results_evaluation.txt"

# ============================================================
# Flag vocabulary (single source of truth)
# ============================================================
RED_FLAG_VOCAB = [
    "pr_not_aligned_with_issue",
    "missing_or_inadequate_test_cases",
    "low_contributor_acceptance_rate",
    "negative_sentiment_in_comments",
    "large_number_of_changed_lines",
    "unclear_or_missing_pr_title",
    "unclear_or_missing_pr_description",
    "unclear_or_missing_commit_messages",
    "poor_code_quality",
    "unresponsive_after_feedback",
    "fails_tests",
    "breaks_compatibility",
    "solution_incorrect_or_inefficient",   # fixed: paper GT had 'solution_is_...' (bug)
    "overengineered_solution",
]
GREEN_FLAG_VOCAB = [
    "clear_alignment_with_issue",
    "includes_test_cases",
    "high_contributor_acceptance_rate",
    "positive_sentiment_in_comments",
    "manageable_number_of_changed_lines",
    "clear_pr_title",
    "clear_pr_description",
    "clear_commit_message",
    "good_code_quality",
    "responsive_to_feedback",
]
ALL_FLAGS = RED_FLAG_VOCAB + GREEN_FLAG_VOCAB
VOCAB_SET = set(ALL_FLAGS)

# ============================================================
# Extractive vs. Evaluative partition (RQ3)
# ============================================================
# Extractive flags — the prompt provides an explicit numeric threshold and
# the model only has to read a value and compare it. No judgment required.
EXTRACTIVE_FLAGS = {
    "low_contributor_acceptance_rate",    # < 50% (from contributor_stats)
    "high_contributor_acceptance_rate",   # >= 80% (from contributor_stats)
    "large_number_of_changed_lines",      # > 500 (from num_changed_lines)
    "manageable_number_of_changed_lines", # < 300 (from num_changed_lines)
}
# Evaluative flags — everything else. Requires semantic judgment about
# code quality, alignment with issue, sentiment, clarity, etc.
EVALUATIVE_FLAGS = set(ALL_FLAGS) - EXTRACTIVE_FLAGS

# Human-label → internal (fixed for solution_*)
H2I_RED = {
    "PR not aligned with issue":            "pr_not_aligned_with_issue",
    "Missing or inadequate test cases":     "missing_or_inadequate_test_cases",
    "Low contributor acceptance rate":      "low_contributor_acceptance_rate",
    "Negative sentiment in comments":       "negative_sentiment_in_comments",
    "Large number of changed lines":        "large_number_of_changed_lines",
    "Unclear or missing PR title":          "unclear_or_missing_pr_title",
    "Unclear or missing PR description":    "unclear_or_missing_pr_description",
    "Unclear or missing commit messages":   "unclear_or_missing_commit_messages",
    "Poor code quality":                    "poor_code_quality",
    "Unresponsive after feedback":          "unresponsive_after_feedback",
    "Fails tests":                          "fails_tests",
    "Risks breaking compatibility":         "breaks_compatibility",
    "Solution is incorrect or inefficient": "solution_incorrect_or_inefficient",  # FIX
    "Overengineered solution":              "overengineered_solution",
}
H2I_GREEN = {
    "Clear alignment with issue":           "clear_alignment_with_issue",
    "Includes test cases":                  "includes_test_cases",
    "High contributor acceptance rate":     "high_contributor_acceptance_rate",
    "Positive sentiment in comments":       "positive_sentiment_in_comments",
    "Manageable number of changed lines":   "manageable_number_of_changed_lines",
    "Clear PR title":                       "clear_pr_title",
    "Clear PR description":                 "clear_pr_description",
    "Clear commit message":                 "clear_commit_message",
    "Good code quality":                    "good_code_quality",
    "Responsive to feedback":               "responsive_to_feedback",
}

# ============================================================
# Prediction file registry
# ============================================================
PRED_FILES = {
    ("GPT-4o-mini",  "zero-shot"):        "batch_zero_shot_results_gpt4o_300.csv",
    ("GPT-4o-mini",  "one-shot-merged"):  "batch_one_shot_results_for_300_gpt4o.csv",
    ("GPT-4o-mini",  "one-shot-unmerged"):"batch_one_shot_results_for_300_gpt4o_unmerged.csv",
    ("GPT-4o-mini",  "few-shot"):         "_GPT_batch_fewSHOT.csv",

    ("DeepSeek R1",  "zero-shot"):        "batch_zero_shot_results_deepseek.csv",   # recovered from prior corruption
    ("DeepSeek R1",  "one-shot-merged"):  "batch_one_shot_results_merged_deepseek.csv",
    ("DeepSeek R1",  "one-shot-unmerged"):"batch_one_shot_results_UNmerged_deepseek.csv",
    ("DeepSeek R1",  "few-shot"):         "_deepseek_batch_fewSHOT.csv",

    ("Mistral 3.2-24B", "zero-shot"):        "mistral_batch_zero_shot_results.csv",
    ("Mistral 3.2-24B", "one-shot-merged"):  "mistral_batch_one_shot_results_merged_mistral.csv",
    ("Mistral 3.2-24B", "one-shot-unmerged"):"mistral_batch_one_shot_results_unmerged_mistral.csv",
    ("Mistral 3.2-24B", "few-shot"):         "_MISTRAL_batch_fewSHOT.csv",

    ("Claude Haiku 4.5", "zero-shot"):        "_HAIKU45_batch_zero_shot_300.csv",
    ("Claude Haiku 4.5", "one-shot-merged"):  "_HAIKU45_batch_one_shot_merged_300.csv",
    ("Claude Haiku 4.5", "one-shot-unmerged"):"_HAIKU45_batch_one_shot_unmerged_300.csv",
    ("Claude Haiku 4.5", "few-shot"):         "_HAIKU45_batch_fewshot_300.csv",
}
MODELS = ["GPT-4o-mini", "Claude Haiku 4.5", "DeepSeek R1", "Mistral 3.2-24B"]
CONDITIONS = ["zero-shot", "one-shot-merged", "one-shot-unmerged", "few-shot"]

# ============================================================
# Ground truth
# ============================================================
def parse_human_flags(cell, mapping):
    if pd.isna(cell) or not str(cell).strip(): return []
    items = [s.strip() for s in str(cell).split(",") if s.strip()]
    return [mapping[s] for s in items if s in mapping]

def pick_final_or_annotator(row, final_col, annot2_col, annot1_col, mapping):
    for c in (final_col, annot2_col, annot1_col):
        v = row.get(c, "")
        if isinstance(v, str) and v.strip():
            return parse_human_flags(v, mapping)
    return []

def load_ground_truth(gt_path=GT_PATH):
    gt = pd.read_csv(gt_path)
    gt = gt[gt["pr_diff"].notna() & (gt["pr_diff"].str.strip() != "")].reset_index(drop=True)
    gt["red_flags_true"] = gt.apply(
        lambda r: pick_final_or_annotator(
            r, "Red Flags (Final)", "Red Flags (Annotator 2)", "Red Flags (Annotator 1)", H2I_RED
        ), axis=1)
    gt["green_flags_true"] = gt.apply(
        lambda r: pick_final_or_annotator(
            r, "Green Flags (Final)", "Green Flags (Annotator 2)", "Green Flags (Annotator 1)", H2I_GREEN
        ), axis=1)
    gt["accepted_true"] = gt["pr_merged"].map({True: "yes", False: "no"})
    gt["gt_index"] = gt.index
    return gt

# ============================================================
# Predictions
# ============================================================
def to_list_safe(cell):
    if not isinstance(cell, str) or not cell.strip(): return []
    try:
        v = ast.literal_eval(cell)
        return v if isinstance(v, list) else []
    except Exception:
        return []

def load_predictions(path):
    """Return DataFrame with columns: index, accepted, red_flags_pred, green_flags_pred, error."""
    try:
        df = pd.read_csv(os.path.join(BASE, path))
    except Exception:
        df = pd.read_csv(os.path.join(BASE, path), engine="python", on_bad_lines="skip")
    df["red_flags_pred"] = df["red_flags"].apply(to_list_safe)
    df["green_flags_pred"] = df["green_flags"].apply(to_list_safe)
    df["index"] = pd.to_numeric(df["index"], errors="coerce").astype("Int64")
    df = df[df["index"].notna()].copy()
    df["index"] = df["index"].astype(int)
    return df

# ============================================================
# Normalization for Mistral (conservative — case + spaces only + known variants)
# ============================================================
_KNOWN_VARIANTS = {
    "clear_commit_messages": "clear_commit_message",         # plural fix
    "unclear_pr_description": "unclear_or_missing_pr_description",
    "unclear_pr_title": "unclear_or_missing_pr_title",
    "unclear_commit_messages": "unclear_or_missing_commit_messages",
    "unclear_commit_message": "unclear_or_missing_commit_messages",
    "solution_is_incorrect_or_inefficient": "solution_incorrect_or_inefficient",
}

def normalize_flag(s):
    """Conservative normalization: lowercase, spaces→underscores, known-variant lookup.
    Returns the canonical vocab string if we can normalize, else None (leave as OOV)."""
    if not isinstance(s, str): return None
    x = s.strip().lower().replace(" ", "_")
    x = re.sub(r"__+", "_", x)  # collapse double underscores
    if x in VOCAB_SET: return x
    if x in _KNOWN_VARIANTS: return _KNOWN_VARIANTS[x]
    return None  # unrecoverable — leave as OOV

def normalize_predictions(flag_list):
    """Return (canonical_list, n_out_of_vocab_before_normalization,
              n_recovered_by_normalization, n_still_oov)."""
    canonical = []
    n_oov_before = 0
    n_recovered = 0
    n_still_oov = 0
    for f in flag_list:
        if f in VOCAB_SET:
            canonical.append(f)
        else:
            n_oov_before += 1
            norm = normalize_flag(f)
            if norm is not None:
                canonical.append(norm)
                n_recovered += 1
            else:
                n_still_oov += 1
    return canonical, n_oov_before, n_recovered, n_still_oov

# ============================================================
# Metric computation
# ============================================================
def compute_metrics(df):
    """
    df must have columns:
      accepted_true, accepted_pred, red_flags_true, green_flags_true,
      red_flags_pred, green_flags_pred
    Rows with NaN accepted_pred are excluded (paper policy).
    Returns dict with all metrics.
    """
    valid = df[df["accepted_pred"].notna()].copy()
    n_total = len(df)
    n_valid = len(valid)
    n_excluded = n_total - n_valid

    if n_valid == 0:
        return None

    # Acceptance
    acc = accuracy_score(valid["accepted_true"], valid["accepted_pred"])
    report = classification_report(
        valid["accepted_true"], valid["accepted_pred"],
        labels=["yes", "no"], output_dict=True, zero_division=0,
    )
    # Bootstrap 95% CI on accuracy
    acc_ci_lo, acc_ci_hi = bootstrap_accuracy_ci(
        valid["accepted_true"].values, valid["accepted_pred"].values
    )

    # Multi-label
    y_true, y_pred = [], []
    for _, r in valid.iterrows():
        t = set(r["red_flags_true"] + r["green_flags_true"])
        p = set(r["red_flags_pred"] + r["green_flags_pred"])
        y_true.append([1 if f in t else 0 for f in ALL_FLAGS])
        y_pred.append([1 if f in p else 0 for f in ALL_FLAGS])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    hloss = hamming_loss(y_true, y_pred)
    flat_acc = float((y_true == y_pred).mean())
    exact_match = float(np.all(y_true == y_pred, axis=1).mean())

    # Per-flag P/R/F1/support
    per_flag = {}
    for i, f in enumerate(ALL_FLAGS):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        per_flag[f] = {
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "support": int(y_true[:, i].sum()),
        }

    return {
        "n_total_pred_rows": int(n_total),
        "n_valid_pred_rows": int(n_valid),
        "n_excluded": int(n_excluded),
        "acceptance_accuracy": float(acc),
        "acceptance_accuracy_ci_lo": acc_ci_lo,
        "acceptance_accuracy_ci_hi": acc_ci_hi,
        "acceptance_p_yes": float(report["yes"]["precision"]),
        "acceptance_r_yes": float(report["yes"]["recall"]),
        "acceptance_f1_yes": float(report["yes"]["f1-score"]),
        "acceptance_p_no": float(report["no"]["precision"]),
        "acceptance_r_no": float(report["no"]["recall"]),
        "acceptance_f1_no": float(report["no"]["f1-score"]),
        "acceptance_macro_avg_f1": float(report["macro avg"]["f1-score"]),
        "acceptance_weighted_avg_f1": float(report["weighted avg"]["f1-score"]),
        "micro_precision": float(p_mi), "micro_recall": float(r_mi), "micro_f1": float(f1_mi),
        "macro_precision": float(p_ma), "macro_recall": float(r_ma), "macro_f1": float(f1_ma),
        "hamming_loss": float(hloss),
        "flattened_accuracy": flat_acc,
        "exact_match_accuracy": exact_match,
        "per_flag": per_flag,
    }

# ============================================================
# DeepSeek zero-shot parser (from paper's .txt)
# ============================================================
def parse_deepseek_zs_txt(path=DS_ZS_TXT):
    """Parse the .txt eval output into the same metrics dict shape."""
    with open(path) as f:
        text = f.read()

    metrics = {"n_total_pred_rows": None, "n_valid_pred_rows": None, "n_excluded": None}

    # Acceptance section
    m = re.search(r"accuracy\s+([\d.]+)\s+(\d+)", text)
    if m:
        metrics["acceptance_accuracy"] = float(m.group(1))
        metrics["n_valid_pred_rows"] = int(m.group(2))
        metrics["n_total_pred_rows"] = 300
        metrics["n_excluded"] = 300 - int(m.group(2))
    for label, key_p, key_r, key_f in [("yes", "acceptance_p_yes","acceptance_r_yes","acceptance_f1_yes"),
                                        ("no",  "acceptance_p_no","acceptance_r_no","acceptance_f1_no")]:
        m = re.search(rf"^\s*{label}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text, re.M)
        if m:
            metrics[key_p] = float(m.group(1))
            metrics[key_r] = float(m.group(2))
            metrics[key_f] = float(m.group(3))
    m = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    if m: metrics["acceptance_macro_avg_f1"] = float(m.group(3))
    m = re.search(r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    if m: metrics["acceptance_weighted_avg_f1"] = float(m.group(3))

    # Overall multi-label
    m = re.search(r"Micro P/R/F1:\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", text)
    if m:
        metrics["micro_precision"] = float(m.group(1))
        metrics["micro_recall"] = float(m.group(2))
        metrics["micro_f1"] = float(m.group(3))
    m = re.search(r"Macro P/R/F1:\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", text)
    if m:
        metrics["macro_precision"] = float(m.group(1))
        metrics["macro_recall"] = float(m.group(2))
        metrics["macro_f1"] = float(m.group(3))
    m = re.search(r"Hamming Loss:\s*([\d.]+)", text)
    if m: metrics["hamming_loss"] = float(m.group(1))
    m = re.search(r"Flattened flag accuracy.*?:\s*([\d.]+)", text)
    if m: metrics["flattened_accuracy"] = float(m.group(1))
    m = re.search(r"Exact-match flag accuracy.*?:\s*([\d.]+)", text)
    if m: metrics["exact_match_accuracy"] = float(m.group(1))

    # Per-flag (only F1 + support easily; use paper's naming with is_ for solution_)
    per_flag = {}
    for line in text.splitlines():
        m = re.match(r"^\d+\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
        if m:
            flag_name = m.group(1)
            # Fix the solution_is_ naming for consistency with corrected vocab
            if flag_name == "solution_is_incorrect_or_inefficient":
                flag_name = "solution_incorrect_or_inefficient"
            per_flag[flag_name] = {
                "precision": float(m.group(2)),
                "recall": float(m.group(3)),
                "f1": float(m.group(4)),
                "support": int(m.group(5)),
            }
    metrics["per_flag"] = per_flag
    return metrics

# ============================================================
# Vocab drift counting for Mistral
# ============================================================
def count_vocab_drift(df):
    """Count how many prediction rows contain ≥1 non-vocab flag (before normalization),
    and total non-vocab emissions."""
    rows_with_oov = 0
    total_oov = 0
    for _, r in df.iterrows():
        flags = r["red_flags_pred"] + r["green_flags_pred"]
        oov = [f for f in flags if f not in VOCAB_SET]
        if oov:
            rows_with_oov += 1
            total_oov += len(oov)
    return rows_with_oov, total_oov

# ============================================================
# Main driver
# ============================================================
def main():
    print("Loading ground truth...")
    gt = load_ground_truth()
    print(f"  GT rows: {len(gt)}")

    acceptance_rows = []
    multilabel_rows = []
    perflag_records = []           # long-format: model, condition, flag, precision, recall, f1, support
    vocab_drift_rows = []
    exclusion_rows = []

    for model in MODELS:
        for cond in CONDITIONS:
            key = (model, cond)
            pred_file = PRED_FILES.get(key)

            print(f"\n=== {model} / {cond} ===")

            # ---- (Formerly special case for DeepSeek zero-shot — now uses recovered CSV) ----
            if False:  # kept structure for reference; DS zero-shot now processed normally
                pass
            else:
                pred = load_predictions(pred_file)
                # Join with GT by index
                df = gt.merge(
                    pred[["index", "accepted", "red_flags_pred", "green_flags_pred"]],
                    left_on="gt_index", right_on="index", how="inner",
                )
                df["accepted_pred"] = df["accepted"]

                # Vocab drift stats (BEFORE normalization) — meaningful for all models
                vocab_rows_oov, vocab_total_oov = count_vocab_drift(df)

                # Strict metrics
                m = compute_metrics(df)

                # Compute normalized metrics for EVERY model (surfaced GPT-4o-mini also drifts in few-shot).
                # For models with no drift, normalized ≈ strict; for drifty models the delta is the finding.
                df_norm = df.copy()
                def norm_row(row_list):
                    canonical, _, _, _ = normalize_predictions(row_list)
                    return canonical
                df_norm["red_flags_pred"]   = df_norm["red_flags_pred"].apply(norm_row)
                df_norm["green_flags_pred"] = df_norm["green_flags_pred"].apply(norm_row)
                m_norm = compute_metrics(df_norm)

                notes = ""
                print(f"  N={m['n_valid_pred_rows']} valid, {m['n_excluded']} excluded, "
                      f"acceptance acc={m['acceptance_accuracy']:.3f}, "
                      f"macro F1={m['macro_f1']:.3f}, vocab drift rows={vocab_rows_oov}")
                if m_norm:
                    print(f"    [normalized] macro F1={m_norm['macro_f1']:.3f} "
                          f"(delta +{m_norm['macro_f1']-m['macro_f1']:.3f})")

            # --- Populate output tables ---
            row = {
                "model": model, "condition": cond,
                "N": m.get("n_valid_pred_rows"), "N_excluded": m.get("n_excluded"),
                "Accuracy":  round(m["acceptance_accuracy"], 3) if "acceptance_accuracy" in m else None,
                "Acc_CI_lo": round(m["acceptance_accuracy_ci_lo"], 3) if "acceptance_accuracy_ci_lo" in m else None,
                "Acc_CI_hi": round(m["acceptance_accuracy_ci_hi"], 3) if "acceptance_accuracy_ci_hi" in m else None,
                "P(Yes)": round(m.get("acceptance_p_yes", np.nan), 3),
                "R(Yes)": round(m.get("acceptance_r_yes", np.nan), 3),
                "F1(Yes)": round(m.get("acceptance_f1_yes", np.nan), 3),
                "P(No)": round(m.get("acceptance_p_no", np.nan), 3),
                "R(No)": round(m.get("acceptance_r_no", np.nan), 3),
                "F1(No)": round(m.get("acceptance_f1_no", np.nan), 3),
                "Macro-Avg F1": round(m.get("acceptance_macro_avg_f1", np.nan), 3),
                "Weighted-Avg F1": round(m.get("acceptance_weighted_avg_f1", np.nan), 3),
                "notes": notes,
            }
            acceptance_rows.append(row)

            multilabel_rows.append({
                "model": model, "condition": cond,
                "N": m.get("n_valid_pred_rows"), "N_excluded": m.get("n_excluded"),
                "Macro-Precision": round(m.get("macro_precision", np.nan), 3),
                "Macro-Recall": round(m.get("macro_recall", np.nan), 3),
                "Macro-F1": round(m.get("macro_f1", np.nan), 3),
                "Micro-Precision": round(m.get("micro_precision", np.nan), 3),
                "Micro-Recall": round(m.get("micro_recall", np.nan), 3),
                "Micro-F1": round(m.get("micro_f1", np.nan), 3),
                "Hamming-Loss": round(m.get("hamming_loss", np.nan), 3),
                "Flattened-Acc": round(m.get("flattened_accuracy", np.nan), 3),
                "Exact-Match-Acc": round(m.get("exact_match_accuracy", np.nan), 3),
                "notes": notes,
            })

            # Per-flag F1
            per_flag = m.get("per_flag", {})
            for flag in ALL_FLAGS:
                if flag in per_flag:
                    perflag_records.append({
                        "model": model, "condition": cond, "flag": flag,
                        "precision": round(per_flag[flag]["precision"], 3),
                        "recall": round(per_flag[flag]["recall"], 3),
                        "f1": round(per_flag[flag]["f1"], 3),
                        "support": per_flag[flag]["support"],
                    })

            # Vocab drift row
            n_valid = m.get("n_valid_pred_rows") or 0
            vocab_drift_rows.append({
                "model": model, "condition": cond,
                "N": n_valid,
                "rows_with_non_vocab_flag": vocab_rows_oov,
                "pct_rows_with_non_vocab_flag": round(100 * vocab_rows_oov / n_valid, 1) if n_valid else 0,
                "total_non_vocab_emissions": vocab_total_oov,
                "strict_macro_F1": round(m.get("macro_f1", np.nan), 3),
                "normalized_macro_F1": round(m_norm["macro_f1"], 3) if m_norm else None,
                "delta_from_normalization": round(m_norm["macro_f1"] - m["macro_f1"], 3) if m_norm else None,
            })

            exclusion_rows.append({
                "model": model, "condition": cond,
                "N_valid": m.get("n_valid_pred_rows"),
                "N_excluded": m.get("n_excluded"),
                "notes": notes,
            })

    # ---- Write output CSVs ----
    def save(rows, name):
        p = os.path.join(BASE, name)
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"  wrote {name}")

    # ---- Majority-class baseline row for Table 5.1 ----
    # Ground truth: gt has accepted_true (yes/no) — count for majority.
    gt_yes = int((gt["accepted_true"] == "yes").sum())
    gt_no = int((gt["accepted_true"] == "no").sum())
    n_gt = gt_yes + gt_no
    majority_label = "yes" if gt_yes >= gt_no else "no"
    # Always-predict-majority metrics
    baseline_acc = max(gt_yes, gt_no) / n_gt
    # P/R/F1 for each class if we always predict majority
    if majority_label == "yes":
        # For "yes" class: TP=gt_yes, FP=gt_no, FN=0 → P=gt_yes/n_gt, R=1.0, F1=2PR/(P+R)
        p_yes = gt_yes / n_gt
        r_yes = 1.0
        f1_yes = 2 * p_yes * r_yes / (p_yes + r_yes) if (p_yes + r_yes) > 0 else 0
        # For "no" class: never predicted → P=undefined(0), R=0, F1=0
        p_no = 0.0; r_no = 0.0; f1_no = 0.0
    else:
        p_no = gt_no / n_gt; r_no = 1.0
        f1_no = 2 * p_no * r_no / (p_no + r_no)
        p_yes = 0.0; r_yes = 0.0; f1_yes = 0.0
    # Bootstrap CI for baseline
    y_true_arr = gt["accepted_true"].values
    y_pred_baseline = np.array([majority_label] * n_gt)
    bl_lo, bl_hi = bootstrap_accuracy_ci(y_true_arr, y_pred_baseline)

    baseline_row = {
        "model": f"Majority baseline (always {majority_label})", "condition": "—",
        "N": n_gt, "N_excluded": 0,
        "Accuracy": round(baseline_acc, 3),
        "Acc_CI_lo": round(bl_lo, 3),
        "Acc_CI_hi": round(bl_hi, 3),
        "P(Yes)": round(p_yes, 3), "R(Yes)": round(r_yes, 3), "F1(Yes)": round(f1_yes, 3),
        "P(No)": round(p_no, 3),  "R(No)": round(r_no, 3),   "F1(No)": round(f1_no, 3),
        "Macro-Avg F1": round((f1_yes + f1_no) / 2, 3),
        "Weighted-Avg F1": round((f1_yes * gt_yes + f1_no * gt_no) / n_gt, 3),
        "notes": f"class balance: {gt_yes} yes / {gt_no} no",
    }
    acceptance_rows.insert(0, baseline_row)   # place at top

    # ---- Extractive vs Evaluative average F1 (RQ3) ----
    extractive_evaluative_rows = []
    perflag_df = pd.DataFrame(perflag_records)
    for model in MODELS:
        for cond in CONDITIONS:
            sub = perflag_df[(perflag_df["model"] == model) & (perflag_df["condition"] == cond)]
            if sub.empty:
                continue
            ext_sub = sub[sub["flag"].isin(EXTRACTIVE_FLAGS)]
            eva_sub = sub[sub["flag"].isin(EVALUATIVE_FLAGS)]
            ext_avg = ext_sub["f1"].mean() if len(ext_sub) else np.nan
            eva_avg = eva_sub["f1"].mean() if len(eva_sub) else np.nan
            extractive_evaluative_rows.append({
                "model": model, "condition": cond,
                "extractive_avg_F1": round(ext_avg, 3) if pd.notna(ext_avg) else None,
                "evaluative_avg_F1": round(eva_avg, 3) if pd.notna(eva_avg) else None,
                "gap": round(ext_avg - eva_avg, 3) if pd.notna(ext_avg) and pd.notna(eva_avg) else None,
                "n_extractive_flags": len(ext_sub),
                "n_evaluative_flags": len(eva_sub),
            })

    print("\n=== Writing output tables ===")
    save(acceptance_rows, "results_table_5_1_acceptance.csv")
    save(multilabel_rows, "results_table_5_2_multilabel.csv")
    save(perflag_records, "results_table_5_3_perflag_all_models.csv")
    save(vocab_drift_rows, "results_table_vocab_drift.csv")
    save(exclusion_rows, "results_exclusion_counts.csv")
    save(extractive_evaluative_rows, "results_table_extractive_vs_evaluative.csv")

    # Pivot per-flag F1 into per-model wide format (24 flags × 4 conditions)
    perflag_df = pd.DataFrame(perflag_records)
    for model in MODELS:
        sub = perflag_df[perflag_df["model"] == model]
        if sub.empty: continue
        pv = sub.pivot(index="flag", columns="condition", values="f1")
        # Ensure vocab order and column order
        pv = pv.reindex(ALL_FLAGS)
        pv = pv.reindex(columns=CONDITIONS)
        pv["Average"] = pv.mean(axis=1).round(3)
        model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model.lower())
        pv.to_csv(os.path.join(BASE, f"results_table_5_perflag_{model_slug}.csv"))
        print(f"  wrote results_table_5_perflag_{model_slug}.csv")

    # Per-model normalized per-flag F1 (paired to strict for direct comparison)
    for model in MODELS:
        if model == "DeepSeek R1":
            # No prediction CSV usable for zero-shot; skip normalized table for DS.
            continue
        norm_records = []
        for cond in CONDITIONS:
            key = (model, cond)
            path = PRED_FILES.get(key)
            if not path: continue
            pred = load_predictions(path)
            df = gt.merge(pred[["index", "accepted", "red_flags_pred", "green_flags_pred"]],
                          left_on="gt_index", right_on="index", how="inner")
            df["accepted_pred"] = df["accepted"]
            df["red_flags_pred"]   = df["red_flags_pred"].apply(lambda L: normalize_predictions(L)[0])
            df["green_flags_pred"] = df["green_flags_pred"].apply(lambda L: normalize_predictions(L)[0])
            m = compute_metrics(df)
            for flag in ALL_FLAGS:
                if flag in m["per_flag"]:
                    norm_records.append({
                        "condition": cond, "flag": flag,
                        "f1": round(m["per_flag"][flag]["f1"], 3),
                    })
        if norm_records:
            model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model.lower())
            df_n = pd.DataFrame(norm_records)
            pv = df_n.pivot(index="flag", columns="condition", values="f1").reindex(ALL_FLAGS).reindex(columns=CONDITIONS)
            pv["Average"] = pv.mean(axis=1).round(3)
            pv.to_csv(os.path.join(BASE, f"results_table_5_perflag_{model_slug}_NORMALIZED.csv"))
            print(f"  wrote results_table_5_perflag_{model_slug}_NORMALIZED.csv")

    print("\nAll tables written. Done.")


if __name__ == "__main__":
    main()
