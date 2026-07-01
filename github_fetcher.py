"""
github_fetcher.py
-----------------
Fetches PR diffs from GitHub API and parses them into
reviewable code chunks.
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """
    Parses a GitHub PR URL into (owner, repo, pr_number).
    Supports:
      https://github.com/owner/repo/pull/123
    """
    pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Invalid GitHub PR URL: {url}")
    owner, repo, pr_number = match.groups()
    return owner, repo, int(pr_number)


def fetch_pr_diff(url: str) -> list[dict]:
    """
    Fetches all changed files from a GitHub PR.
    Returns a list of dicts with keys:
      - filename: str
      - language: str  (inferred from extension)
      - chunks: list[str]  (individual diff hunks)
      - additions: int
      - deletions: int
    """
    if not GITHUB_TOKEN:
        raise EnvironmentError(
            "GITHUB_TOKEN not found. Make sure your .env file exists and "
            "contains GITHUB_TOKEN=your_token"
        )

    owner, repo, pr_number = parse_pr_url(url)

    # Fetch list of files changed in the PR
    files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    response = requests.get(files_url, headers=HEADERS)

    if response.status_code == 401:
        raise PermissionError("GitHub token is invalid or expired.")
    if response.status_code == 404:
        raise FileNotFoundError(f"PR not found: {url}")
    if response.status_code != 200:
        raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")

    files = response.json()
    parsed_files = []

    for f in files:
        filename = f.get("filename", "")
        patch = f.get("patch", "")

        if not patch:
            continue  # binary files, renames with no content change, etc.

        chunks = split_diff_into_chunks(patch)
        parsed_files.append({
            "filename": filename,
            "language": infer_language(filename),
            "chunks": chunks,
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
        })

    if not parsed_files:
        raise ValueError("No reviewable code changes found in this PR.")

    return parsed_files


def split_diff_into_chunks(patch: str, max_lines: int = 30) -> list[str]:
    """
    Splits a raw diff patch into chunks of max_lines lines each.
    This keeps individual inputs to the model manageable.
    """
    lines = patch.splitlines()
    chunks = []
    current_chunk = []

    for line in lines:
        current_chunk.append(line)
        if len(current_chunk) >= max_lines:
            chunks.append("\n".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def infer_language(filename: str) -> str:
    """Infer programming language from file extension."""
    ext_map = {
        ".py":   "Python",
        ".js":   "JavaScript",
        ".ts":   "TypeScript",
        ".java": "Java",
        ".cpp":  "C++",
        ".c":    "C",
        ".go":   "Go",
        ".rs":   "Rust",
        ".rb":   "Ruby",
        ".cs":   "C#",
        ".php":  "PHP",
        ".kt":   "Kotlin",
        ".swift":"Swift",
        ".sh":   "Shell",
        ".sql":  "SQL",
        ".html": "HTML",
        ".css":  "CSS",
    }
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext_map.get(ext, "Unknown")


def chunk_raw_code(code: str, max_lines: int = 30) -> list[dict]:
    """
    Takes raw pasted code and splits it into chunks for review.
    Returns same format as fetch_pr_diff for consistency.
    """
    lines = code.splitlines()
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunks.append("\n".join(lines[i:i + max_lines]))

    return [{
        "filename": "pasted_code",
        "language": "Unknown",
        "chunks": chunks,
        "additions": len(lines),
        "deletions": 0,
    }]
