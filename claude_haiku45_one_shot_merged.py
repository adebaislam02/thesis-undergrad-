# -*- coding: utf-8 -*-
"""
Claude Haiku 4.5 — one-shot MERGED example, full run.

Byte-identical one-shot prompt template to one_shot_gpt.py (GPT-4o-mini),
oneshot_merged_deepseek.py, and oneshot_merged_mistral.py. Only intentional
deltas from those scripts:
  - model slug (anthropic/claude-haiku-4.5)
  - inline retry with exponential backoff
  - Unicode surrogate stripping in text cleaning
  - output CSV keyed by (pr_url, index) composite so PR-issue pair
    duplicates are both preserved (same behavior as other prev models
    save for the composite key)

The in-context example is dataset row index 7 (rails/rails/pull/50983) —
the same example used across all three previous models' one-shot merged
scripts. That row is excluded from the eval set (matches prev models),
yielding 299 output rows.
"""

import requests
import pandas as pd
import json
import re
import ast
import os
import time
from tqdm import tqdm

# --- CONSTANTS ---
API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "anthropic/claude-haiku-4.5"
INPUT_FILE = "/Users/adeba/Downloads/FINAL_DATASET_FIXED.csv"
OUTPUT_FILE = "_HAIKU45_batch_one_shot_merged_300.csv"

EXAMPLE_INDEX = 7  # rails/rails/pull/50983 — same merged example as GPT/DeepSeek/Mistral

MAX_RETRIES = 5
BASE_DELAY = 1.0

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


# --- HELPERS ---
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def remove_surrogates(txt):
    if not isinstance(txt, str):
        return txt
    return _SURROGATE_RE.sub("", txt)


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Preserve pr_url — used for output keying (delta from prev scripts which drop it).
    cols_to_drop = [
        "Unnamed: 0",
        "linked_issue_links",
        "Green Flags (Annotator 2)",
        "Red Flags (Annotator 2)",
        "Total Score (Annotator 2)",
        "Notes (Annotator 2)",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df[df["pr_diff"].notna() & (df["pr_diff"].str.strip() != "")].reset_index(drop=True)

    def clean_text_field(txt):
        if not isinstance(txt, str):
            txt = "" if pd.isna(txt) else str(txt)
        t = remove_surrogates(txt)
        t = re.sub(r"``````", "", t)
        t = re.sub(r"\\begin\{code\}[\s\S]*?\\end\{code\}", "", t)
        t = re.sub(r"```.*", "", t)
        t = re.sub(r"\\begin\{code\}", "", t)
        t = re.sub(r"\\end\{code\}", "", t)
        t = t.strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def clean_diff(raw_diff):
        if not isinstance(raw_diff, str):
            return ""
        raw_diff = remove_surrogates(raw_diff)
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

    def parse_list_and_join(cell, sep="\n\n"):
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

    def parse_and_join_flags(cell):
        if not isinstance(cell, str) or cell.strip() == "":
            return ""
        parts = [p.strip() for p in cell.split(",")]
        return ", ".join(p for p in parts if p)

    def parse_contributor_stats(cell):
        if not isinstance(cell, str) or cell.strip() == "":
            return {}
        try:
            return json.loads(cell)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(cell)
            except Exception:
                return {}

    def contributor_stats_to_string(stats):
        if not stats:
            return ""
        return "; ".join(f"{k}: {v}" for k, v in stats.items())

    def get_final_or_fallback(row, final_col, fallback_col):
        final_val = row.get(final_col, "")
        fallback_val = row.get(fallback_col, "")
        return final_val if isinstance(final_val, str) and final_val.strip() else fallback_val

    for col in ["pr_commit_messages", "pr_comments", "pr_review_comments"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_list_and_join(x, sep="\n\n"))

    # Clean annotator 1 raw flag columns first
    for flag_col in ["Green Flags (Annotator 1)", "Red Flags (Annotator 1)"]:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].apply(parse_and_join_flags)

    # Compute Final-or-fallback columns (used only when the row is the in-context example)
    if "Green Flags (Final)" in df.columns and "Green Flags (Annotator 1)" in df.columns:
        df["Green Flags FF"] = df.apply(
            lambda row: parse_and_join_flags(
                get_final_or_fallback(row, "Green Flags (Final)", "Green Flags (Annotator 1)")
            ),
            axis=1,
        )
    elif "Green Flags (Annotator 1)" in df.columns:
        df["Green Flags FF"] = df["Green Flags (Annotator 1)"]

    if "Red Flags (Final)" in df.columns and "Red Flags (Annotator 1)" in df.columns:
        df["Red Flags FF"] = df.apply(
            lambda row: parse_and_join_flags(
                get_final_or_fallback(row, "Red Flags (Final)", "Red Flags (Annotator 1)")
            ),
            axis=1,
        )
    elif "Red Flags (Annotator 1)" in df.columns:
        df["Red Flags FF"] = df["Red Flags (Annotator 1)"]

    if "Notes (Final)" in df.columns and "Notes (Annotator 1)" in df.columns:
        df["Notes FF"] = df.apply(
            lambda row: clean_text_field(
                get_final_or_fallback(row, "Notes (Final)", "Notes (Annotator 1)")
            ),
            axis=1,
        )
    elif "Notes (Annotator 1)" in df.columns:
        df["Notes FF"] = df["Notes (Annotator 1)"].apply(clean_text_field)

    if "Total Score (Final)" in df.columns and "Total Score (Annotator 1)" in df.columns:
        df["Total Score FF"] = df.apply(
            lambda row: get_final_or_fallback(row, "Total Score (Final)", "Total Score (Annotator 1)"),
            axis=1,
        )
    elif "Total Score (Annotator 1)" in df.columns:
        df["Total Score FF"] = df["Total Score (Annotator 1)"]

    for col in ["title", "body", "issue_titles", "issue_description", "contributor_stats"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_field)

    df["pr_diff"] = df["pr_diff"].apply(clean_diff)

    if "num_changed_lines" in df.columns:
        df["num_changed_lines"] = (
            pd.to_numeric(df["num_changed_lines"], errors="coerce").fillna(0).astype(int)
        )

    if "pr_merged" in df.columns:
        df["pr_merged"] = df["pr_merged"].apply(
            lambda x: True if str(x).strip().lower() in {"true", "yes", "1"} else False
        )

    if "contributor_stats" in df.columns:
        df["contributor_stats"] = (
            df["contributor_stats"].apply(parse_contributor_stats).apply(contributor_stats_to_string)
        )

    return df


def get_notes_for_example(row):
    notes_ff = row.get("Notes FF", "")
    notes1 = row.get("Notes (Annotator 1)", "")
    if isinstance(notes_ff, str) and notes_ff.strip():
        return notes_ff.strip()
    elif isinstance(notes1, str) and notes1.strip():
        return notes1.strip()
    else:
        return ""


def render_pr_block(row, use_notes=True, is_example=False):
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


def build_one_shot_prompt(df, example_index, target_row):
    example = df.iloc[example_index]
    ex_block = render_pr_block(example, use_notes=True, is_example=True)

    ex_acc = example.get("Total Score FF", 0)
    try:
        ex_acc = "yes" if float(ex_acc) >= 0.5 else "no"
    except (ValueError, TypeError):
        ex_acc = "no"

    ex_red = [s.strip() for s in example["Red Flags FF"].split(",") if s.strip()] if isinstance(example["Red Flags FF"], str) else []
    ex_green = [s.strip() for s in example["Green Flags FF"].split(",") if s.strip()] if isinstance(example["Green Flags FF"], str) else []

    ex_answer = json.dumps({
        "accepted": ex_acc,
        "red_flags": ex_red,
        "green_flags": ex_green
    })

    tgt_block = render_pr_block(target_row, use_notes=False)

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
    - pr_aligned_with_issue: The PR fully addresses the issue or solves the issue with some unrelated changes that do not harm the solution.
    - pr_not_aligned_with_issue: The pull request does not address or solve the linked issue, or its content is unrelated.
    # Here is a worked example. Copy this structure exactly:
    Example Input:
    {ex_block}

    Example Output:
    {ex_answer}
    ---
    Now process this new PR:
    {tgt_block}

    Answer with only the JSON object starting at '{{' and ending at '}}':'''
    return template


def extract_json(raw_text):
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
    snippet = raw_text[start:end + 1]
    snippet = snippet.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return ast.literal_eval(snippet)


def call_haiku_with_retries(prompt, api_key, model=MODEL_NAME, max_tokens=1024, max_retries=MAX_RETRIES):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                return raw, {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            elif resp.status_code == 429 or resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                wait = BASE_DELAY * (2 ** attempt)
                print(f"[retry {attempt+1}/{max_retries}] {resp.status_code} — sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except (requests.exceptions.RequestException, UnicodeEncodeError) as e:
            last_err = e
            wait = BASE_DELAY * (2 ** attempt)
            print(f"[retry {attempt+1}/{max_retries}] {type(e).__name__}: {e} — sleeping {wait:.1f}s")
            time.sleep(wait)
    raise last_err if last_err else RuntimeError("Unknown failure")


def batch_process_one_shot(df, example_index, output_path, api_key):
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set — put it in .env and re-run.")

    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        processed = set(zip(prev["pr_url"], prev["index"].astype(int)))
        records = prev.to_dict("records")
        print(f"Resuming: {len(processed)} (pr_url, index) pairs already processed")
    else:
        processed = set()
        records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Haiku-4.5 one-shot merged"):
        if idx == example_index:
            continue  # skip the in-context example row (matches GPT/DeepSeek/Mistral)
        pr_url = row["pr_url"]
        if (pr_url, int(idx)) in processed:
            continue

        prompt = build_one_shot_prompt(df, example_index, row)

        result = {
            "pr_url": pr_url,
            "index": int(idx),
            "accepted": None,
            "red_flags": str([]),
            "green_flags": str([]),
            "error": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        try:
            raw, usage = call_haiku_with_retries(prompt, api_key)
            result["prompt_tokens"] = usage.get("prompt_tokens")
            result["completion_tokens"] = usage.get("completion_tokens")
            result["total_tokens"] = usage.get("total_tokens")

            parsed = extract_json(raw)
            red = parsed.get("red_flags", [])
            if red == ["(none)"]:
                red = []
            green = parsed.get("green_flags", [])
            if green == ["(none)"]:
                green = []
            result["accepted"] = parsed.get("accepted")
            result["red_flags"] = str(red)
            result["green_flags"] = str(green)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            print(f"[ERROR @ pr_url={pr_url} idx={idx}] {result['error']}")

        records.append(result)
        pd.DataFrame(records).drop_duplicates(subset=["pr_url", "index"]).to_csv(output_path, index=False)

    print(f"\nOne-shot merged run complete. Results saved to: {output_path}")
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_and_preprocess_data(INPUT_FILE)
    print(f"Loaded dataset: {len(df)} rows  (in-context example row idx={EXAMPLE_INDEX} → {df.iloc[EXAMPLE_INDEX]['pr_url']})")
    results = batch_process_one_shot(df, EXAMPLE_INDEX, OUTPUT_FILE, API_KEY)
    print("\nSample results:")
    print(results[["pr_url", "index", "accepted", "error"]].head())
