"""
relabel_with_gemini.py
------------------------
Re-labels the CRAVE-derived training data using Gemini as a judge,
replacing the noisy keyword-based pseudo-labels with LLM-derived labels.

Why this should help:
  - Keyword matching on reviewer explanation text was producing near-random
    accuracy (31%) when used to fine-tune CodeBERT — the labels themselves
    didn't correlate with actual code patterns
  - An LLM reading the code diff directly and reasoning about category is a
    fundamentally stronger labelling signal than string matching on
    unrelated explanation text
  - This is still NOT ground truth (no human verification), but it should
    be meaningfully better than the previous approach

Features:
  - Checkpointing: saves progress every 20 rows so a crash/rate-limit doesn't
    lose all progress — safe to stop and resume
  - Retry logic: handles transient API errors and rate limits gracefully
  - Confidence tracking: keeps Gemini's self-reported confidence so you can
    filter to only "high confidence" rows for training if needed
  - Validation: flags rows where Gemini's response didn't parse cleanly,
    so you can review/discard them rather than silently corrupting labels

Usage:
    pip install google-generativeai pandas tqdm
    python relabel_with_gemini.py
"""

import google.generativeai as genai
import pandas as pd
import time
import re
import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

INPUT_CSV       = "training_data_full.csv"
OUTPUT_CSV      = "training_data_gemini_labelled.csv"
CHECKPOINT_EVERY = 20
MAX_CODE_CHARS  = 1500       # truncate long diffs to keep prompt reasonable
RATE_LIMIT_SLEEP = 1.0       # seconds between calls — adjust per your API tier
MAX_RETRIES     = 3

CATEGORIES = {"bug", "security", "performance", "style", "ok"}

PROMPT_TEMPLATE = """You are an expert code reviewer. Classify this code diff into EXACTLY ONE of these 5 categories based on what it primarily represents:

- bug: Fixes incorrect logic, crashes, null/undefined handling, edge cases, off-by-one errors, or any code that was producing wrong behavior
- security: Addresses vulnerabilities, injection risks, unsafe authentication, hardcoded credentials, unsanitized input, or other security weaknesses
- performance: Improves speed, reduces memory usage, removes redundant computation, optimizes loops/algorithms, or addresses scalability
- style: Improves naming, formatting, readability, comments, documentation, or code organization without changing behavior
- ok: The diff has no significant issues - it's a clean addition/change with nothing notably wrong (new feature, routine update, etc.)

Respond in EXACTLY this format, nothing else, no explanation:
LABEL: <category>
CONFIDENCE: <high/medium/low>

If the diff doesn't clearly fit one category, pick the closest match and mark confidence as low.

Here is the diff:

{code}
"""


# ── Gemini setup ────────────────────────────────────────────────────────────

def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file or set it as "
            "an environment variable. Get a free key at https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_response(text: str) -> tuple[str | None, str | None]:
    """
    Parses Gemini's response into (label, confidence).
    Returns (None, None) if parsing fails — caller should flag these rows.
    """
    label_match = re.search(r"LABEL:\s*(\w+)", text, re.IGNORECASE)
    conf_match  = re.search(r"CONFIDENCE:\s*(\w+)", text, re.IGNORECASE)

    if not label_match:
        return None, None

    label = label_match.group(1).lower().strip()
    confidence = conf_match.group(1).lower().strip() if conf_match else "unknown"

    if label not in CATEGORIES:
        return None, None

    return label, confidence


# ── Single-row labelling with retries ─────────────────────────────────────────

def label_one_row(model, code: str) -> tuple[str | None, str | None]:
    prompt = PROMPT_TEMPLATE.format(code=code[:MAX_CODE_CHARS])

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            label, confidence = parse_response(response.text)

            if label is not None:
                return label, confidence

            # Parsing failed but API call succeeded — don't retry, just flag it
            return None, None

        except Exception as e:
            wait = (attempt + 1) * 5
            print(f"\n  [Attempt {attempt+1}/{MAX_RETRIES}] Error: {e}")
            print(f"  Retrying in {wait}s …")
            time.sleep(wait)

    return None, None  # all retries exhausted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset …")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["code"]).reset_index(drop=True)
    print(f"  {len(df)} rows to label\n")

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    if Path(OUTPUT_CSV).exists():
        existing = pd.read_csv(OUTPUT_CSV)
        already_done = existing["gemini_label"].notna().sum()
        print(f"Found existing checkpoint with {already_done} rows already labelled.")
        print("Resuming from where it left off …\n")
        df = existing
    else:
        df["gemini_label"]      = None
        df["gemini_confidence"] = None

    model = setup_gemini()

    failed_rows = []

    rows_to_process = df[df["gemini_label"].isna()].index.tolist()
    print(f"Rows remaining: {len(rows_to_process)}\n")

    for count, idx in enumerate(tqdm(rows_to_process, desc="Labelling")):
        code = df.at[idx, "code"]
        label, confidence = label_one_row(model, code)

        if label is None:
            failed_rows.append(idx)
            df.at[idx, "gemini_label"]      = "PARSE_FAILED"
            df.at[idx, "gemini_confidence"] = "none"
        else:
            df.at[idx, "gemini_label"]      = label
            df.at[idx, "gemini_confidence"] = confidence

        time.sleep(RATE_LIMIT_SLEEP)

        # ── Checkpoint periodically ───────────────────────────────────────────
        if (count + 1) % CHECKPOINT_EVERY == 0:
            df.to_csv(OUTPUT_CSV, index=False)

    # Final save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved final labelled dataset -> {OUTPUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    valid = df[df["gemini_label"] != "PARSE_FAILED"]
    print(f"\nSuccessfully labelled: {len(valid)} / {len(df)}")
    print(f"Failed to parse:       {len(failed_rows)}")

    print("\nNew label distribution:")
    print(valid["gemini_label"].value_counts())

    print("\nConfidence distribution:")
    print(valid["gemini_confidence"].value_counts())

    print("\nLabel agreement with original keyword-based labels:")
    agreement = (valid["gemini_label"] == valid["label"]).mean()
    print(f"  {agreement*100:.1f}% of rows got the same label as before")
    print("  (Low agreement is expected and OK — that's the point of relabeling)")

    if failed_rows:
        print(f"\nWarning: {len(failed_rows)} rows failed to parse and were marked "
              f"'PARSE_FAILED'. Consider dropping these before fine-tuning:")
        print(f"  df = df[df['gemini_label'] != 'PARSE_FAILED']")


if __name__ == "__main__":
    main()
