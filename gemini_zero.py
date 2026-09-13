# -*- coding: utf-8 -*-

import requests
import pandas as pd
import json
import re
import ast
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, hamming_loss

API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

file_path = "FINAL_DATASET - 300_Annotated.csv"

# 1) Read data CSV
df = pd.read_csv(file_path)
print("First 5 rows:\n", df.head())
print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing counts per column:\n", df.isnull().sum())

def load_and_preprocess_data(filepath):
  

    # 1) Load data
    df = pd.read_csv(filepath)

    # 2) Drop unwanted columns
    cols_to_drop = [
        "Unnamed: 0",
        "pr_url",
        "linked_issue_links",
        "Green Flags (Annotator 2)",
        "Red Flags (Annotator 2)",
        "Total Score (Annotator 2)",
        "Notes (Annotator 2)",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 3) Filter empty diffs
    df = df[df["pr_diff"].notna() & (df["pr_diff"].str.strip() != "")].reset_index(drop=True)

    # Cleaning functions (as before)
    def clean_text_field(txt: str) -> str:
        if not isinstance(txt, str):
            txt = "" if pd.isna(txt) else str(txt)
        t = txt
        t = re.sub(r"``````", "", t)
        t = re.sub(r"\\begin\{code\}[\s\S]*?\\end\{code\}", "", t)
        t = re.sub(r"```.*", "", t)
        t = re.sub(r"\\begin\{code\}", "", t)
        t = re.sub(r"\\end\{code\}", "", t)
        t = t.strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def clean_diff(raw_diff: str) -> str:
        if not isinstance(raw_diff, str):
            return ""
        cleaned_lines = []
        for line in raw_diff.splitlines():
            if re.match(r"\s*```", line):
                continue
            if re.match(r"\s*\\begin\{code\}", line) or re.match(r"\s*\\end\{code\}", line):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def parse_list_and_join(cell: str, sep="\n\n") -> str:
        if not isinstance(cell, str) or cell.strip() == "":
            return ""
        try:
            lst = ast.literal_eval(cell)
            if isinstance(lst, list):
                cleaned_elems = [clean_text_field(str(item)) for item in lst if str(item).strip()]
                return sep.join(cleaned_elems)
        except Exception:
            return clean_text_field(cell)
        return ""

    def parse_and_join_flags(cell: str) -> str:
        if not isinstance(cell, str) or cell.strip() == "":
            return ""
        parts = [p.strip() for p in cell.split(",")]
        return ", ".join(p for p in parts if p)

    def parse_contributor_stats(cell: str) -> dict:
        if not isinstance(cell, str) or cell.strip() == "":
            return {}
        try:
            return json.loads(cell)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(cell)
            except Exception:
                return {}

    def contributor_stats_to_string(stats: dict) -> str:
        if not stats:
            return ""
        parts = []
        for k, v in stats.items():
            parts.append(f"{k}: {v}")
        return "; ".join(parts)

    def get_final_or_fallback(row, final_col, fallback_col):
        final_val = row.get(final_col, "")
        fallback_val = row.get(fallback_col, "")
        return final_val if isinstance(final_val, str) and final_val.strip() else fallback_val

    # 5) Parsing operations for list-type columns
    for col in ["pr_commit_messages", "pr_comments", "pr_review_comments"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_list_and_join(x, sep="\n\n"))

    # Clean annotator 1 flags columns first
    for flag_col in ["Green Flags (Annotator 1)", "Red Flags (Annotator 1)"]:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].apply(parse_and_join_flags)

    # 6) Create unified final-or-fallback columns with 'FF' suffix
    if "Green Flags (Final)" in df.columns and "Green Flags (Annotator 1)" in df.columns:
        df["Green Flags FF"] = df.apply(
            lambda row: parse_and_join_flags(get_final_or_fallback(row, "Green Flags (Final)", "Green Flags (Annotator 1)")),
            axis=1
        )
    elif "Green Flags (Annotator 1)" in df.columns:
        df["Green Flags FF"] = df["Green Flags (Annotator 1)"]

    if "Red Flags (Final)" in df.columns and "Red Flags (Annotator 1)" in df.columns:
        df["Red Flags FF"] = df.apply(
            lambda row: parse_and_join_flags(get_final_or_fallback(row, "Red Flags (Final)", "Red Flags (Annotator 1)")),
            axis=1
        )
    elif "Red Flags (Annotator 1)" in df.columns:
        df["Red Flags FF"] = df["Red Flags (Annotator 1)"]

    if "Notes (Final)" in df.columns and "Notes (Annotator 1)" in df.columns:
        df["Notes FF"] = df.apply(
            lambda row: clean_text_field(get_final_or_fallback(row, "Notes (Final)", "Notes (Annotator 1)")),
            axis=1
        )
    elif "Notes (Annotator 1)" in df.columns:
        df["Notes FF"] = df["Notes (Annotator 1)"].apply(clean_text_field)

    if "Total Score (Final)" in df.columns and "Total Score (Annotator 1)" in df.columns:
        df["Total Score FF"] = df.apply(
            lambda row: get_final_or_fallback(row, "Total Score (Final)", "Total Score (Annotator 1)"),
            axis=1
        )
    elif "Total Score (Annotator 1)" in df.columns:
        df["Total Score FF"] = df["Total Score (Annotator 1)"]

    # Clean main text fields (note: cleaning of "Notes FF" already done above)
    text_cols = [
        "title",
        "body",
        "issue_titles",
        "issue_description",
        "contributor_stats",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_field)

    # Clean diffs
    df["pr_diff"] = df["pr_diff"].apply(clean_diff)

    # Convert numeric to int
    if "num_changed_lines" in df.columns:
        df["num_changed_lines"] = (
            pd.to_numeric(df["num_changed_lines"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # Convert pr_merged to boolean
    if "pr_merged" in df.columns:
        df["pr_merged"] = df["pr_merged"].apply(
            lambda x: True if str(x).strip().lower() in {"true", "yes", "1"} else False
        )

    # Process contributor stats JSON
    if "contributor_stats" in df.columns:
        df["contributor_stats"] = (
            df["contributor_stats"]
            .apply(parse_contributor_stats)
            .apply(contributor_stats_to_string)
        )

    return df


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
    "solution_incorrect_or_inefficient",
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

RED_FLAGS_LINE = ", ".join(RED_FLAG_VOCAB)
GREEN_FLAGS_LINE = ", ".join(GREEN_FLAG_VOCAB)


def get_notes_for_example(row: pd.Series) -> str:
    notes2 = row.get("Notes", "")
    notes1 = row.get("Notes (Annotator 1)", "")
    if isinstance(notes2, str) and notes2.strip():
        return notes2.strip()
    elif isinstance(notes1, str) and notes1.strip():
        return notes1.strip()
    else:
        return ""


def call_gemini_2_flash(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
    }
    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            content = response_json["choices"][0].get("message", {}).get("content", "")
            if content:
                return content
            else:
                print("Empty content field returned in the response.")
                return None
        else:
            print("Error: 'choices' not found or empty in the API response.")
            return None
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def render_pr_block(row: pd.Series, use_notes: bool = True, is_example: bool = False) -> str:
    def or_none(x):
        if not isinstance(x, str) or x.strip() == "":
            return "(none)"
        return x.strip()

    lines = [
        f"PR Title: {or_none(row.get('title', ''))}",
        f"Description: {or_none(row.get('body', ''))}",
        "Commit Messages:",
        or_none(row.get("pr_commit_messages", "")),
        "",
        "PR Comments:",
        or_none(row.get("pr_comments", "")),
        "",
        "Review Comments:",
        or_none(row.get("pr_review_comments", "")),
        "Code Diff:",
        or_none(row.get("pr_diff", "")),
        "",
        f"Number of Changed Lines: {row.get('num_changed_lines', 0)}",
        f"Contributor Stats: {or_none(row.get('contributor_stats', ''))}",
        f"Issue Title: {or_none(row.get('issue_titles', ''))}",
        f"Issue Description: {or_none(row.get('issue_description', ''))}",
    ]

    if use_notes and is_example:
        notes = get_notes_for_example(row)
        lines.append(f"Annotator Notes: {notes if notes else '(none)'}")

    return "\n".join(lines)


def build_zero_shot_prompt(row: pd.Series) -> str:
    pr_block = render_pr_block(row)

    template = f'''Instruction:
You are a pull-request review assistant. Given PR metadata, first predict acceptance ("yes" or "no"), then select which flags apply.
Return valid JSON with exactly three keys: "accepted", "red_flags", and "green_flags"—no extra text.

Allowed red_flags: {RED_FLAGS_LINE}
Allowed green_flags: {GREEN_FLAGS_LINE}

Flag definitions:
- low_contributor_acceptance_rate: Contributor's historical PR acceptance rate is less than 50% (see "contributor stats" field).
- high_contributor_acceptance_rate: Contributor's acceptance rate is 80% or higher.
- manageable_number_of_changed_lines: The number of changed lines is less than 300.
- large_number_of_changed_lines: The PR changes more than 500 lines.
- clear_alignment_with_issue: The PR fully addresses the issue or solves the issue with some unrelated changes that do not harm the solution.
- pr_not_aligned_with_issue: The pull request does not address or solve the linked issue, or its content is unrelated.

Now process this PR:
{pr_block}

Answer with only the JSON object starting at '{{' and ending at '}}':'''
    return template


def extract_json(raw_text: str) -> dict:
    start = raw_text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    depth = 0
    end = None
    for i, ch in enumerate(raw_text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise ValueError("Unmatched braces in model output")

    snippet = raw_text[start: end + 1]
    snippet = snippet.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return ast.literal_eval(snippet)


def batch_process_zero_shot(df: pd.DataFrame, output_path: str = 'batch_zero_shot_results_gemini.csv') -> pd.DataFrame:
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        processed_indices = set(prev['index'])
        records = prev.to_dict('records')
        print(f"Resuming from {len(processed_indices)} previously processed PRs.")
    else:
        processed_indices = set()
        records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Zero-Shot Batch"):
        if idx in processed_indices:
            continue

        prompt = build_zero_shot_prompt(row)

        try:
            raw = call_gemini_2_flash(prompt)
            if raw:
                parsed = extract_json(raw)
                result = {
                    "index": idx,
                    "accepted": parsed.get("accepted"),
                    "red_flags": parsed.get("red_flags", []),
                    "green_flags": parsed.get("green_flags", []),
                    "error": None,
                }
            else:
                result = {
                    "index": idx,
                    "accepted": None,
                    "red_flags": [],
                    "green_flags": [],
                    "error": "API Call failed",
                }
        except Exception as e:
            print(f"[ERROR @ idx={idx}] {e}")
            result = {
                "index": idx,
                "accepted": None,
                "red_flags": [],
                "green_flags": [],
                "error": str(e),
            }

        records.append(result)

        if len(records) % 1 == 0 or idx == len(df) - 1:
            pd.DataFrame(records).to_csv(output_path, index=False)
            print(f"Processed PR index {idx} - results saved to {output_path}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_and_preprocess_data(file_path)

    output_path = 'batch_zero_shot_results_gemini_300.csv'
    results_df = batch_process_zero_shot(df, output_path=output_path)
    print(f"Results saved to {output_path}")
    print(results_df.head())
