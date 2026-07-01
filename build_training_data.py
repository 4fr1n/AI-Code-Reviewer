"""
build_training_data.py
-----------------------
Loads the CRAVE dataset, uses keyword matching on the human-written
`explanation` field to derive bug / style / performance / security / ok
labels, and saves a clean CSV ready for fine-tuning CodeBERT.

Run this locally (not in the sandbox — needs internet access to HF Hub):
    pip install datasets pandas
    python build_training_data.py
"""

from datasets import load_dataset
import pandas as pd
import re

# ── Keyword patterns per category ─────────────────────────────────────────────
# These run against the `explanation` field (human-written reviewer reasoning)
# to derive a pseudo-label for each diff.

PATTERNS = {
    "security": [
        r"\bsecurity\b", r"\bvulnerab\w*\b", r"\binjection\b", r"\bsanitiz\w*\b",
        r"\bauth\w*\b", r"\bcredential\w*\b", r"\bpassword\b", r"\btoken\b",
        r"\bexploit\b", r"\bunsafe\b", r"\bcve\b", r"\bxss\b",
    ],
    "bug": [
        r"\bbug\b", r"\bfix(?:es|ed)?\b", r"\berror\b", r"\bcrash\w*\b",
        r"\bfail\w*\b", r"\bincorrect\b", r"\bwrong\b", r"\bbroken\b",
        r"\bnull\b", r"\bundefined\b", r"\bedge case\b", r"\boff[- ]by[- ]one\b",
        r"\bregression\b", r"\bdoes not (?:work|handle)\b",
    ],
    "performance": [
        r"\bperformance\b", r"\bslow\b", r"\befficien\w*\b", r"\boptimiz\w*\b",
        r"\bmemory leak\b", r"\blatency\b", r"\bbottleneck\b", r"\bredundant\b",
        r"\bcomplexity\b", r"\bo\(n", r"\bcache\b", r"\bthroughput\b",
    ],
    "style": [
        r"\bstyle\b", r"\bnaming\b", r"\bconvention\w*\b", r"\breadab\w*\b",
        r"\bformat\w*\b", r"\btypo\b", r"\bdocstring\b", r"\bcomment\w*\b",
        r"\blint\w*\b", r"\bconsisten\w*\b", r"\bclean\s?up\b",
    ],
}

CATEGORY_PRIORITY = ["security", "bug", "performance", "style"]  # checked in this order


def derive_label(explanation: str, original_label: str) -> str:
    """
    Returns one of: bug / style / performance / security / ok

    Logic:
      - If original CRAVE label is APPROVE -> likely 'ok' (no issues raised)
      - If REQUEST_CHANGES -> scan explanation text for category keywords
        in priority order (security checked first since it's rarest/most critical)
      - If REQUEST_CHANGES but no keywords match -> default to 'bug'
        (most review change-requests are bug-related if not otherwise specified)
    """
    if original_label == "APPROVE":
        return "ok"

    text = explanation.lower()

    for category in CATEGORY_PRIORITY:
        for pattern in PATTERNS[category]:
            if re.search(pattern, text):
                return category

    return "bug"  # default fallback for unmatched REQUEST_CHANGES


def truncate_patch(patch: str, max_chars: int = 2000) -> str:
    """Truncate long patches to keep CodeBERT input manageable (512 tokens)."""
    return patch[:max_chars]


def main():
    print("Loading CRAVE dataset from HuggingFace Hub …")
    dataset = load_dataset("TuringEnterprises/CRAVE", split="train")
    print(f"  Loaded {len(dataset)} rows\n")

    rows = []
    for example in dataset:
        label = derive_label(example["explanation"], example["label"])
        code = truncate_patch(example["patch"])

        rows.append({
            "code": code,
            "label": label,
            "original_label": example["label"],
            "repo": example["repo"],
            "pr_number": example["pr_number"],
        })

    df = pd.DataFrame(rows)

    print("Label distribution:")
    print(df["label"].value_counts())
    print()

    # Save full dataset
    df.to_csv("training_data_full.csv", index=False)
    print("Saved training_data_full.csv")

    # Print a few examples per category for sanity checking
    print("\n" + "=" * 60)
    print("SAMPLE CHECK — verify these labels look reasonable")
    print("=" * 60)
    for cat in ["bug", "style", "performance", "security", "ok"]:
        subset = df[df["label"] == cat]
        if len(subset) > 0:
            print(f"\n--- {cat.upper()} (n={len(subset)}) ---")
            sample = subset.iloc[0]
            print(f"Repo: {sample['repo']}")
            print(f"Code snippet: {sample['code'][:200]}...")


if __name__ == "__main__":
    main()
