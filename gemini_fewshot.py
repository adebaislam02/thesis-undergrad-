


import pandas as pd
import json
import requests
import re
import ast
import os
from tqdm import tqdm

# --- CONSTANTS ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.0-flash-001"
INPUT_FILE = "FINAL_DATASET - 300_Annotated.csv"

OUTPUT_FILE = "_GEMINI_FEWSHOT.csv"
EXAMPLE_INDICES = [88, 140, 261]

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

# --- HELPER FUNCTIONS ---
def clean_text_field(txt):
    if not isinstance(txt, str):
        txt = "" if pd.isna(txt) else str(txt)
    t = txt
    t = re.sub(r"``````", "", t)           # FIXED typo in regex
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
    # FIX: Make sure row is a dict-like object, never a list!
    # If it's a Series, .get() works
    if isinstance(row, dict) or isinstance(row, pd.Series):
        final_val = row.get(final_col, "")
        fallback_val = row.get(fallback_col, "")
        return final_val if isinstance(final_val, str) and final_val.strip() else fallback_val
    # Otherwise (should never happen), fallback to index-based
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
        or_none(row.get("pr_commit_messages","")),
        "",
        "PR Comments:",
        or_none(row.get("pr_comments","")),
        "",
        "Review Comments:",
        or_none(row.get("pr_review_comments","")),
        "Code Diff:",
        or_none(row.get("pr_diff","")),
        "",
        f"Number of Changed Lines: {row.get('num_changed_lines','')}",
        f"Contributor Stats: {or_none(row.get('contributor_stats',''))}",
        f"Issue Title: {or_none(row.get('issue_titles',''))}",
        f"Issue Description: {or_none(row.get('issue_description',''))}",
    ]
    if include_notes and "Notes" in row:
        lines.append(f"Annotator Notes: {or_none(row['Notes'])}")
    return "\n".join(lines)

def build_few_shot_prompt(df, example_indices, target_row):
    try:
        examples = []
        for example_index in example_indices:
            example = df.iloc[example_index]    # This returns a Series!
            ex_block = render_pr_block(example, include_notes=True)
            ex_acc = example.get("Total Score", 0)
            ex_acc = "yes" if float(ex_acc) >= 0.5 else "no"
            ex_red = example.get("Red Flags", "")
            ex_red = [s.strip() for s in ex_red.split(",") if s.strip()] if isinstance(ex_red, str) else []
            ex_green = example.get("Green Flags", "")
            ex_green = [s.strip() for s in ex_green.split(",") if s.strip()] if isinstance(ex_green, str) else []
            if not ex_red:
                ex_red = ["(none)"]
            if not ex_green:
                ex_green = ["(none)"]
            ex_answer = json.dumps({
                "accepted": ex_acc,
                "red_flags": ex_red,
                "green_flags": ex_green
            }, indent=2)
            examples.append(f"Example Input:\n{ex_block}\nExample Output:\n{ex_answer}\n")
        
        # Now include the target row's PR information in the prompt
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

Answer with only the JSON object starting at '{{' and ending at '}}':'''
        
     
        return template
    except Exception as e:
        print("Error while building few-shot prompt:", e)
        return None


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
    snippet = raw_text[start : end+1]
    snippet = snippet.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return ast.literal_eval(snippet)

def process_via_openrouter(prompt, api_key, model=MODEL_NAME, max_tokens=512):
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ],
    "max_tokens": max_tokens
}

    response = requests.post(endpoint, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        # FIX: Should index into first result in "choices"
        raw_text = data["choices"][0]["message"]["content"]
        return raw_text
    else:
        print(f"[API ERROR {response.status_code}]: {response.text}")
        return None

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    # Drop unwanted columns if they exist
    cols_to_drop = [
        "Unnamed: 0", "pr_url", "linked_issue_links",
        "Green Flags (Annotator 2)", "Red Flags (Annotator 2)", "Total Score (Annotator 2)", "Notes (Annotator 2)"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    df = df[df["pr_diff"].notna() & (df["pr_diff"].str.strip() != "")].reset_index(drop=True)
    # Text fields
    for col in ["pr_commit_messages", "pr_comments", "pr_review_comments"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_list_and_join(x, sep="\n\n"))
    df["Green Flags"] = df.apply(
        lambda row: parse_and_join_flags(get_final_or_fallback(row, "Green Flags (Final)", "Green Flags (Annotator 1)")),
        axis=1
    )
    df["Red Flags"] = df.apply(
        lambda row: parse_and_join_flags(get_final_or_fallback(row, "Red Flags (Final)", "Red Flags (Annotator 1)")),
        axis=1
    )
    df["Notes"] = df.apply(
        lambda row: clean_text_field(get_final_or_fallback(row, "Notes (Final)", "Notes (Annotator 1)")),
        axis=1
    ) if "Notes (Final)" in df.columns and "Notes (Annotator 1)" in df.columns else ""
    df["Total Score"] = df.apply(
        lambda row: get_final_or_fallback(row, "Total Score (Final)", "Total Score (Annotator 1)"),
        axis=1
    ) if "Total Score (Final)" in df.columns and "Total Score (Annotator 1)" in df.columns else ""
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

from tqdm import tqdm
import pandas as pd
import os

def batch_process_few_shot(
    df: pd.DataFrame, 
    example_indices: list, 
    output_path: str = "batch_few_shot_results_prinfo_code.csv",
    api_key: str = None
) -> pd.DataFrame:
    # Resume from previous results if output_path exists
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        processed_indices = set(prev['index'])
        records = prev.to_dict('records')
        print(f"Resuming from {len(processed_indices)} previously processed PRs.")
    else:
        processed_indices = set()
        records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Few-Shot Batch"):
        if idx in example_indices or idx in processed_indices:
            continue  # Skip example PRs and already processed PRs

        prompt = build_few_shot_prompt(df, example_indices, row)

        try:
            raw = process_via_openrouter(prompt, api_key)
            if raw:
                parsed = extract_json(raw)

                # Update red_flags to be [] instead of ['(none)'] if no red flags
                red_flags = parsed.get("red_flags", [])
                if red_flags == ["(none)"]:
                    red_flags = []

                result = {
                    "index": idx,
                    "accepted": parsed.get("accepted"),
                    "red_flags": str(red_flags),   # Store empty list as []
                    "green_flags": str(parsed.get("green_flags", [])),
                    "error": None,
                }
            else:
                result = {
                    "index": idx,
                    "accepted": None,
                    "red_flags": str([]),   # Python style empty list
                    "green_flags": str([]),
                    "error": "API Call failed",
                }
        except Exception as e:
            print(f"[ERROR @ idx={idx}] {e}")
            result = {
                "index": idx,
                "accepted": None,
                "red_flags": str([]),  # Python style empty list
                "green_flags": str([]),
                "error": str(e),
            }

        records.append(result)

        # Save after every PR
        pd.DataFrame(records).drop_duplicates(subset=["index"]).to_csv(output_path, index=False)
        print(f"Processed PR index {idx} - results saved to {output_path}")

    print(f"\n✓ Final results saved to: {output_path}")

    return pd.DataFrame(records)



# # --- MAIN ---
# if __name__ == "__main__":
#     df = load_and_preprocess_data(INPUT_FILE)
#     results_df = batch_process_few_shot(df, EXAMPLE_INDICES, OUTPUT_FILE, OPENROUTER_API_KEY)
#     print("\nSample results:")
#     print(results_df.head())
def rerun_failed_prs(
    df: pd.DataFrame,
    failed_indices: list, 
    output_file: str,
    api_key: str = None,
    example_indices: list = EXAMPLE_INDICES
):
    records = []
    for idx in failed_indices:
        if idx not in df.index:
            print(f"Warning: index {idx} not in DataFrame")
            continue
        row = df.loc[idx]
        prompt = build_few_shot_prompt(df, example_indices, row)
        try:
            raw = process_via_openrouter(prompt, api_key)
            if raw:
                parsed = extract_json(raw)
                red_flags = parsed.get("red_flags", [])
                if red_flags == ["(none)"]:
                    red_flags = []
                result = {
                    "index": idx,
                    "accepted": parsed.get("accepted"),
                    "red_flags": str(red_flags),
                    "green_flags": str(parsed.get("green_flags", [])),
                    "error": None,
                }
            else:
                result = {
                    "index": idx,
                    "accepted": None,
                    "red_flags": str([]),
                    "green_flags": str([]),
                    "error": "API Call failed",
                }
        except Exception as e:
            result = {
                "index": idx,
                "accepted": None,
                "red_flags": str([]),
                "green_flags": str([]),
                "error": str(e)
            }
        records.append(result)

        # Save after each PR rerun to avoid loss
        pd.DataFrame(records).drop_duplicates(subset=["index"]).to_csv(output_file, index=False)
        print(f"Rerun PR index {idx} saved to {output_file}")

    print(f"Rerun complete; results saved to {output_file}")
    return pd.DataFrame(records)


# --- MAIN ---
def main():
    # Load data
    df = load_and_preprocess_data(INPUT_FILE)

    # Run your main few-shot batch process
    results_df = batch_process_few_shot(df, EXAMPLE_INDICES, OUTPUT_FILE, OPENROUTER_API_KEY)
    print("\nSample results:")
    print(results_df.head())

    # Suppose you maintain a list of failed PR indices
    failed_indices = [150,151,152,154,155,158,161,164,168,170,171, 178,180,185,187,193,196, 203,209, 222,231, 236,239, 272,273,274,278,279, 281,285,286,288,289,294,124]  # Replace with your actual failed PR indices

    # Rerun failed PRs and save separately
    if failed_indices:
        rerun_output_file = "_GEMINI_FEWSHOT_RERUN_failed.csv"
        rerun_results = rerun_failed_prs(df, failed_indices, rerun_output_file, OPENROUTER_API_KEY, EXAMPLE_INDICES)
        print("\nRerun failed PRs results:")
        print(rerun_results.head())


if __name__ == "__main__":
    main()

