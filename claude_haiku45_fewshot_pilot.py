"""
Claude Sonnet 5 — few-shot PILOT run (25 PRs).

Diagnostic pilot: identical few-shot prompt construction to few_shot_gpt.py,
plus a `justifications` field (per-flag) so we can manually verify the model
is actually engaging with PR content vs. surface-cue pattern-matching.

Full runs will not include the `justifications` field and will match the
other three models' output schema exactly.
"""

import pandas as pd
import json
import re
import ast
import os
import random
import time
from tqdm import tqdm
from openai import OpenAI

# --- CONSTANTS ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = "anthropic/claude-haiku-4.5"
INPUT_FILE = "/Users/adeba/Downloads/FINAL_DATASET_FIXED.csv"
OUTPUT_FILE = "haiku45_fewshot_PILOT_with_justification.csv"

# Few-shot example PRs — pinned by pr_url (indices 88, 140, 261 in the fixed dataset)
EXAMPLE_PR_URLS = [
    "https://github.com/huggingface/transformers/pull/12514",
    "https://github.com/eslint/eslint/pull/14702",
    "https://github.com/pallets/flask/pull/5544",
]

PILOT_SAMPLE_SIZE = 25
PILOT_SEED = 42

# Retry config
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


# --- HELPER FUNCTIONS (copied verbatim from few_shot_gpt.py) ---
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def remove_surrogates(txt):
    if not isinstance(txt, str):
        return txt
    return _SURROGATE_RE.sub("", txt)


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
        if re.match(r"\s*```", line) or re.match(r"\s*\\begin\{code\}", line) or re.match(r"\s*\\end\{code\}", line):
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
        return ""
    try:
        stats = json.loads(cell)
    except json.JSONDecodeError:
        try:
            stats = ast.literal_eval(cell)
        except Exception:
            return ""
    if isinstance(stats, dict):
        parts = [f"{k}: {v}" for k, v in stats.items()]
        return "; ".join(parts)
    return ""


def get_final_or_fallback(row, final_col, fallback_col):
    if isinstance(row, dict) or isinstance(row, pd.Series):
        final_val = row.get(final_col, "")
        fallback_val = row.get(fallback_col, "")
        return final_val if isinstance(final_val, str) and final_val.strip() else fallback_val
    return ""


def render_pr_block(row, include_notes=False):
    def or_none(x):
        if not isinstance(x, str) or x.strip() == "":
            return "(none)"
        return x
    lines = [
        f"PR Title: {or_none(row.get('title',''))}",
        f"Description: {or_none(row.get('body',''))}",
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
        f"Number of Changed Lines: {row.get('num_changed_lines','')}",
        f"Contributor Stats: {or_none(row.get('contributor_stats',''))}",
        f"Issue Title: {or_none(row.get('issue_titles',''))}",
        f"Issue Description: {or_none(row.get('issue_description',''))}",
    ]
    if include_notes and "Notes" in row:
        lines.append(f"Annotator Notes: {or_none(row['Notes'])}")
    return "\n".join(lines)


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    # PILOT CHANGE: keep pr_url — we key output by pr_url. All other preprocessing identical.
    cols_to_drop = [
        "Unnamed: 0", "linked_issue_links",
        "Green Flags (Annotator 2)", "Red Flags (Annotator 2)",
        "Total Score (Annotator 2)", "Notes (Annotator 2)",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    df = df[df["pr_diff"].notna() & (df["pr_diff"].str.strip() != "")].reset_index(drop=True)
    for col in ["pr_commit_messages", "pr_comments", "pr_review_comments"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_list_and_join(x, sep="\n\n"))
    df["Green Flags"] = df.apply(
        lambda row: parse_and_join_flags(get_final_or_fallback(row, "Green Flags (Final)", "Green Flags (Annotator 1)")),
        axis=1,
    )
    df["Red Flags"] = df.apply(
        lambda row: parse_and_join_flags(get_final_or_fallback(row, "Red Flags (Final)", "Red Flags (Annotator 1)")),
        axis=1,
    )
    df["Notes"] = df.apply(
        lambda row: clean_text_field(get_final_or_fallback(row, "Notes (Final)", "Notes (Annotator 1)")),
        axis=1,
    ) if "Notes (Final)" in df.columns or "Notes (Annotator 1)" in df.columns else ""
    df["Total Score"] = df.apply(
        lambda row: get_final_or_fallback(row, "Total Score (Final)", "Total Score (Annotator 1)"),
        axis=1,
    ) if "Total Score (Final)" in df.columns or "Total Score (Annotator 1)" in df.columns else ""
    for col in ["title", "body", "issue_titles", "issue_description", "contributor_stats"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_field)
    df["pr_diff"] = df["pr_diff"].apply(clean_diff)
    if "num_changed_lines" in df.columns:
        df["num_changed_lines"] = pd.to_numeric(df["num_changed_lines"], errors="coerce").fillna(0).astype(int)
    if "pr_merged" in df.columns:
        df["pr_merged"] = df["pr_merged"].apply(lambda x: True if str(x).strip().lower() in {"true", "yes", "1"} else False)
    if "contributor_stats" in df.columns:
        df["contributor_stats"] = df["contributor_stats"].apply(parse_contributor_stats)
    return df


def build_few_shot_prompt_with_justification(df, example_urls, target_row):
    """
    Few-shot prompt: byte-identical to few_shot_gpt.py's build_few_shot_prompt(),
    with a pilot-only trailing instruction that requires a `justifications` field.
    """
    examples = []
    for ex_url in example_urls:
        example = df[df["pr_url"] == ex_url].iloc[0]
        ex_block = render_pr_block(example, include_notes=True)
        ex_acc = example.get("Total Score", 0)
        try:
            ex_acc = "yes" if float(ex_acc) >= 0.5 else "no"
        except (ValueError, TypeError):
            ex_acc = "no"
        ex_red = example.get("Red Flags", "")
        ex_red = [s.strip() for s in ex_red.split(",") if s.strip()] if isinstance(ex_red, str) else []
        ex_green = example.get("Green Flags", "")
        ex_green = [s.strip() for s in ex_green.split(",") if s.strip()] if isinstance(ex_green, str) else []
        if not ex_red:
            ex_red = ["(none)"]
        if not ex_green:
            ex_green = ["(none)"]
        ex_answer = json.dumps(
            {"accepted": ex_acc, "red_flags": ex_red, "green_flags": ex_green},
            indent=2,
        )
        examples.append(f"Example Input:\n{ex_block}\nExample Output:\n{ex_answer}\n")

    tgt_block = render_pr_block(target_row, include_notes=False)
    separator = "\n---\n"
    joined_examples = separator.join(examples)

    template = f'''Instruction:
You are a pull-request review assistant. Given PR metadata and code diff, first predict acceptance ("yes" or "no"), then select which flags apply.
Return valid JSON with exactly three keys: "accepted", "red_flags", and "green_flags"—no extra text.

Allowed red_flags: {RED_FLAGS_LINE}
Allowed green_flags: {GREEN_FLAGS_LINE}

flag definitions:
    - low_contributor_acceptance_rate: Contributor's historical PR acceptance rate is less than 50% (see "contributor stats" field).
    - high_contributor_acceptance_rate: Contributor's acceptance rate is 80% or higher.
    - manageable_number_of_changed_lines: The number of changed lines is less than 300.
    - large_number_of_changed_lines: The PR changes more than 500 lines.
    - pr_aligned_with_issue: The PR fully addresses the issue or solves the issue with some unrelated changes that do not harm the solution.
    - pr_not_aligned_with_issue: The pull request does not address or solve the linked issue, or its content is unrelated.
Here are some worked examples. Copy this style exactly:
{joined_examples}
---
Now process this new PR:
{tgt_block}

ADDITIONAL PILOT INSTRUCTION: In addition to the three keys above, include a fourth key "justifications" — a JSON object mapping each flag name you output (from both red_flags and green_flags) to a one-sentence explanation grounded in this specific PR's content (its diff, comments, issue, or contributor stats). Do not include justifications for flags you did not select.

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


def call_sonnet5_with_retries(client, prompt, model=MODEL_NAME, max_tokens=2048, max_retries=MAX_RETRIES):
    """Inline retry with exponential backoff. Returns (raw_text, usage_dict)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            raw = completion.choices[0].message.content
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", None) if completion.usage else None,
                "completion_tokens": getattr(completion.usage, "completion_tokens", None) if completion.usage else None,
                "total_tokens": getattr(completion.usage, "total_tokens", None) if completion.usage else None,
            }
            return raw, usage
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            is_rate = "429" in msg or "rate" in msg or "overloaded" in msg
            wait = BASE_DELAY * (2 ** attempt)
            print(f"[retry {attempt+1}/{max_retries}] {'rate-limit' if is_rate else 'error'}: {e} — sleeping {wait:.1f}s")
            time.sleep(wait)
    raise last_err


def batch_pilot(df, example_urls, output_path, api_key, n_pilot=PILOT_SAMPLE_SIZE, seed=PILOT_SEED):
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set — put it in .env and re-run.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Candidate pool: all PRs except the 3 few-shot example PRs
    candidates = df[~df["pr_url"].isin(example_urls)].copy()
    print(f"Candidate pool size (excluding 3 few-shot examples): {len(candidates)}")

    # Deterministic random sample
    random.seed(seed)
    pilot_pr_urls = random.sample(candidates["pr_url"].tolist(), n_pilot)
    print(f"Sampled {len(pilot_pr_urls)} pilot PRs (seed={seed})")

    # Resume support
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        processed = set(prev["pr_url"])
        records = prev.to_dict("records")
        print(f"Resuming: {len(processed)} PRs already processed")
    else:
        processed = set()
        records = []

    for pr_url in tqdm(pilot_pr_urls, desc="Sonnet-5 few-shot pilot"):
        if pr_url in processed:
            continue

        row_matches = df[df["pr_url"] == pr_url]
        if row_matches.empty:
            continue
        row = row_matches.iloc[0]
        idx = int(row_matches.index[0])

        prompt = build_few_shot_prompt_with_justification(df, example_urls, row)

        result = {
            "pr_url": pr_url,
            "index": idx,
            "accepted": None,
            "red_flags": str([]),
            "green_flags": str([]),
            "justifications": None,
            "error": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        try:
            raw, usage = call_sonnet5_with_retries(client, prompt)
            result["prompt_tokens"] = usage.get("prompt_tokens")
            result["completion_tokens"] = usage.get("completion_tokens")
            result["total_tokens"] = usage.get("total_tokens")

            if raw is None:
                result["error"] = "empty API response"
            else:
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
                result["justifications"] = json.dumps(parsed.get("justifications", {}), ensure_ascii=False)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            print(f"[ERROR @ pr_url={pr_url}] {result['error']}")

        records.append(result)
        pd.DataFrame(records).drop_duplicates(subset=["pr_url"]).to_csv(output_path, index=False)

    print(f"\nPilot complete. Results saved to: {output_path}")
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_and_preprocess_data(INPUT_FILE)
    print(f"Loaded dataset: {len(df)} rows")
    results = batch_pilot(df, EXAMPLE_PR_URLS, OUTPUT_FILE, OPENROUTER_API_KEY)
    print("\nSample results:")
    print(results[["pr_url", "accepted", "error"]].head())
