# -*- coding: utf-8 -*-
"""
Recover parser-skipped rows in the four affected prediction CSVs, and
merge DeepSeek rerun files into their main counterparts.

Does NOT touch: DeepSeek zero-shot (skipped per user request — keeping
paper's existing results as-is).

Every file gets backed up as `<name>.pre_recovery_backup` before rewrite.
"""

import pandas as pd
import ast
import re
import os
import shutil

BASE = "/Users/adeba/thesis-undergrad-"


def parse_broken_line(line):
    """
    Parse a single raw CSV data line that may be malformed with unquoted list literals.

    Expected shape:  index,accepted,red_list,green_list,[error]

    - Fields 3 (red_list) and 4 (green_list) are Python list literals.
    - Lists may be wrapped in outer double quotes ("[...]") or not ([...]).
    - Lists may contain commas + single-quoted or double-quoted string elements.
    """
    line = line.rstrip("\n\r")
    if not line.strip():
        return None

    # First two fields are simple: index, accepted
    parts = line.split(",", 2)
    if len(parts) < 3:
        return None
    try:
        idx = int(parts[0])
    except ValueError:
        return None
    accepted = parts[1].strip().strip('"').strip("'")
    if accepted not in ("yes", "no", ""):
        return None
    rest = parts[2]

    # Now find two bracket-balanced list literals in `rest`.
    lists = []
    i = 0
    while i < len(rest) and len(lists) < 2:
        # skip whitespace, commas, and stray quote wrappers
        while i < len(rest) and rest[i] in ' ",':
            i += 1
        if i >= len(rest):
            break
        if rest[i] != "[":
            # Not a list here — maybe empty field. Advance.
            # Look for next comma.
            next_comma = rest.find(",", i)
            if next_comma == -1:
                break
            i = next_comma + 1
            lists.append("[]")
            continue

        # Bracket-balanced scan, respecting string literals inside
        depth = 0
        j = i
        in_str = False
        str_char = None
        while j < len(rest):
            c = rest[j]
            if in_str:
                if c == "\\" and j + 1 < len(rest):
                    j += 2
                    continue
                if c == str_char:
                    in_str = False
            else:
                if c in ("'", '"'):
                    in_str = True
                    str_char = c
                elif c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        lists.append(rest[i:j + 1])
                        i = j + 1
                        break
            j += 1
        else:
            # Reached end without closing — malformed beyond recovery
            return None

    # Parse recovered list strings
    def try_eval(s):
        s = s.strip().strip('"')
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        return []

    red = try_eval(lists[0]) if len(lists) >= 1 else []
    green = try_eval(lists[1]) if len(lists) >= 2 else []

    return {
        "index": idx,
        "accepted": accepted if accepted else None,
        "red_flags": str(red),
        "green_flags": str(green),
        "error": None,
    }


def load_with_recovery(path):
    """
    Load a CSV that may have some malformed lines. Try standard parsing first,
    then recover skipped lines with parse_broken_line().
    Returns (clean_df, recovered_indexes list).
    """
    # Standard parse — skip bad lines
    df_clean = pd.read_csv(path, engine="python", on_bad_lines="skip")
    ok_indexes = set(pd.to_numeric(df_clean["index"], errors="coerce").dropna().astype(int).tolist())

    # Find which raw lines were skipped
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    recovered = []
    recovered_idx = []
    for line in raw_lines[1:]:  # skip header
        # Try to detect this is a data line starting with a digit
        m = re.match(r"^\s*(\d+)\s*,", line)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in ok_indexes:
            continue  # already loaded fine
        parsed = parse_broken_line(line)
        if parsed and parsed["index"] not in ok_indexes:
            recovered.append(parsed)
            recovered_idx.append(parsed["index"])
            ok_indexes.add(parsed["index"])

    if recovered:
        df_rec = pd.DataFrame(recovered)
        # Align columns
        for c in df_clean.columns:
            if c not in df_rec.columns:
                df_rec[c] = None
        df_rec = df_rec[df_clean.columns]
        df_out = pd.concat([df_clean, df_rec], ignore_index=True)
    else:
        df_out = df_clean

    df_out = df_out.sort_values("index").reset_index(drop=True)
    return df_out, recovered_idx


def merge_rerun(main_df, rerun_path):
    """Merge rerun rows into main_df — rerun wins on matching index."""
    rerun_df = pd.read_csv(rerun_path, engine="python", on_bad_lines="skip")
    rerun_idx = set(pd.to_numeric(rerun_df["index"], errors="coerce").dropna().astype(int).tolist())
    # Ensure columns match
    for c in main_df.columns:
        if c not in rerun_df.columns:
            rerun_df[c] = None
    rerun_df = rerun_df[main_df.columns]

    # Drop main rows that will be overwritten by rerun
    main_kept = main_df[~pd.to_numeric(main_df["index"], errors="coerce").isin(rerun_idx)]
    merged = pd.concat([main_kept, rerun_df], ignore_index=True)
    merged = merged.sort_values("index").reset_index(drop=True)
    return merged, sorted(rerun_idx)


# ============================================================
# Configuration of what to fix
# ============================================================
JOBS = [
    {
        "label": "Mistral one-shot MERGED",
        "path": os.path.join(BASE, "mistral_batch_one_shot_results_merged_mistral.csv"),
        "rerun": None,
        "expected_indexes": set(range(300)) - {7},
    },
    {
        "label": "Mistral one-shot UNMERGED",
        "path": os.path.join(BASE, "mistral_batch_one_shot_results_unmerged_mistral.csv"),
        "rerun": None,
        "expected_indexes": set(range(300)) - {215},
    },
    {
        "label": "DeepSeek one-shot MERGED",
        "path": os.path.join(BASE, "batch_one_shot_results_merged_deepseek.csv"),
        "rerun": os.path.join(BASE, "RERUN_failed_one_shot.csv"),
        "expected_indexes": set(range(300)) - {7},
    },
    {
        "label": "DeepSeek one-shot UNMERGED",
        "path": os.path.join(BASE, "batch_one_shot_results_UNmerged_deepseek.csv"),
        "rerun": os.path.join(BASE, "DEpseek_fled_one_shot.csv"),
        "expected_indexes": set(range(300)) - {215},
    },
]


def process_job(job):
    path = job["path"]
    label = job["label"]
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    if not os.path.exists(path):
        print(f"  !! Main file missing: {path}")
        return

    # Backup once (don't overwrite existing backup)
    backup = path + ".pre_recovery_backup"
    if not os.path.exists(backup):
        shutil.copy(path, backup)
        print(f"  Backed up: {os.path.basename(backup)}")
    else:
        print(f"  Backup already exists: {os.path.basename(backup)} (kept)")

    # Step 1: robust parse + line recovery
    df, recovered_idx = load_with_recovery(path)
    print(f"  Loaded {len(df)} rows after recovery")
    if recovered_idx:
        print(f"  Recovered from malformed lines: idx {recovered_idx}")
    else:
        print(f"  No malformed lines needed recovery")

    # Step 2: merge rerun if any
    if job["rerun"] and os.path.exists(job["rerun"]):
        df, merged_idx = merge_rerun(df, job["rerun"])
        print(f"  Merged rerun ({os.path.basename(job['rerun'])}): overwrote idx {merged_idx}")

    # Step 3: check remaining gaps
    present = set(pd.to_numeric(df["index"], errors="coerce").dropna().astype(int).tolist())
    missing = sorted(job["expected_indexes"] - present)
    if missing:
        print(f"  !! Still missing after recovery + merge: idx {missing}")
    else:
        print(f"  Complete: all {len(job['expected_indexes'])} expected indexes present")

    # Step 4: check null accepted (rows that ran but returned no answer)
    nulls = int(df["accepted"].isna().sum())
    print(f"  Rows with null accepted: {nulls}")

    # Step 5: rewrite CSV cleanly with proper quoting
    import csv
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"  Rewrote: {os.path.basename(path)} ({len(df)} rows, clean quoting)")


if __name__ == "__main__":
    for j in JOBS:
        process_job(j)
    print("\nDone.")
