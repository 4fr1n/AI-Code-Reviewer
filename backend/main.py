"""
main.py — FastAPI backend for the Code Review Assistant
---------------------------------------------------------
Exposes the existing reviewer_model.py + github_fetcher.py pipeline
as a REST API for the React frontend.

Run with:
    uvicorn main:app --reload --port 8000
"""

import sys
import os
import time
import asyncio
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path so we can import the existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from github_fetcher import fetch_pr_diff, chunk_raw_code
from reviewer_model import CodeReviewer

# ── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Code Review Assistant API",
    description="CodeBERT classification + Llama feedback generation",
    version="1.0.0",
)

# Allow the React dev server to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup, not per-request ──────────────────────────────
# This is important — loading CodeBERT takes several seconds, we don't want
# that happening on every API call.

reviewer: CodeReviewer | None = None


@app.on_event("startup")
async def load_models():
    global reviewer
    print("Loading models …")
    reviewer = CodeReviewer()
    print("Models loaded, API ready.")


# ── Request / Response schemas ─────────────────────────────────────────────────

class ReviewPRRequest(BaseModel):
    pr_url: str = Field(..., description="GitHub PR URL to review")


class ReviewCodeRequest(BaseModel):
    code: str = Field(..., description="Raw code to review")
    filename: str = Field(default="pasted_code", description="Optional filename for display")


class Finding(BaseModel):
    category: Literal["bug", "style", "performance", "security", "ok"]
    confidence: float
    feedback: str
    chunk: str
    chunk_index: int
    filename: str
    language: str


class FileResult(BaseModel):
    filename: str
    language: str
    additions: int
    deletions: int
    findings: list[Finding]


class ReviewResponse(BaseModel):
    files: list[FileResult]
    total_chunks: int
    total_issues: int
    elapsed_seconds: float
    category_counts: dict[str, int]


# ── Core review logic (shared between endpoints) ─────────────────────────────

async def run_review(files: list[dict]) -> ReviewResponse:
    if reviewer is None:
        raise HTTPException(status_code=503, detail="Models still loading, try again shortly.")

    start = time.time()
    file_results = []
    total_chunks = 0
    category_counts = {"bug": 0, "style": 0, "performance": 0, "security": 0, "ok": 0}

    for file_info in files:
        # Run the blocking model inference in a thread so we don't block
        # the FastAPI event loop
        results = await asyncio.to_thread(reviewer.review_file, file_info)

        findings = []
        for r in results:
            category_counts[r["category"]] += 1
            total_chunks += 1
            findings.append(Finding(
                category=r["category"],
                confidence=r["confidence"],
                feedback=r["feedback"],
                chunk=r["chunk"],
                chunk_index=r["chunk_index"],
                filename=r["filename"],
                language=r["language"],
            ))

        file_results.append(FileResult(
            filename=file_info["filename"],
            language=file_info["language"],
            additions=file_info["additions"],
            deletions=file_info["deletions"],
            findings=findings,
        ))

    total_issues = sum(v for k, v in category_counts.items() if k != "ok")

    return ReviewResponse(
        files=file_results,
        total_chunks=total_chunks,
        total_issues=total_issues,
        elapsed_seconds=round(time.time() - start, 1),
        category_counts=category_counts,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok" if reviewer is not None else "loading",
        "models_loaded": reviewer is not None,
    }


@app.post("/review/pr", response_model=ReviewResponse)
async def review_pr(request: ReviewPRRequest):
    """Review all files changed in a GitHub PR."""
    try:
        files = fetch_pr_diff(request.pr_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch PR: {e}")

    return await run_review(files)


@app.post("/review/code", response_model=ReviewResponse)
async def review_code(request: ReviewCodeRequest):
    """Review raw pasted code."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty.")

    files = chunk_raw_code(request.code)
    files[0]["filename"] = request.filename

    return await run_review(files)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
