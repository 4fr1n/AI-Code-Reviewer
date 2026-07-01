# AI Code Review Assistant
### Fine-tuned CodeBERT · Llama 3.2 · FastAPI · React

An end-to-end AI-powered code review tool that classifies GitHub PR diffs and pasted code snippets into bug / security / performance / style / clean categories using a fine-tuned CodeBERT model, then generates natural-language review comments using a locally-hosted Llama 3.2 model.

---

## Demo

![Code Review Assistant UI](demo/screenshot.png)

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Model Training](#2-model-training)
   - 2.1 [Dataset](#21-dataset)
   - 2.2 [Labelling Pipeline](#22-labelling-pipeline)
   - 2.3 [Fine-tuning CodeBERT](#23-fine-tuning-codebert)
   - 2.4 [Results](#24-results)
3. [Backend — FastAPI](#3-backend--fastapi)
4. [Frontend — React](#4-frontend--react)
5. [Running Locally](#5-running-locally)
6. [Project Structure](#6-project-structure)

---

## 1. Architecture Overview

The system is a two-stage NLP pipeline:

**Stage 1 — Classification (CodeBERT)**
Each code chunk is passed through a fine-tuned CodeBERT model which produces a category label and confidence score. CodeBERT is a transformer pretrained on both natural language and code (GitHub repositories), making it well-suited for understanding code semantics.

**Stage 2 — Feedback Generation (Llama 3.2)**
The category label and code chunk are combined into a structured prompt and sent to a locally-hosted Llama 3.2:1b model via Ollama. The model generates a concise, actionable review comment specific to the identified issue type.

```
GitHub PR URL / Pasted Code
         │
         ▼
  GitHub Fetcher (github_fetcher.py)
  - Fetches PR diff via GitHub REST API
  - Splits diff into 30-line chunks
  - Infers language from file extension
         │
         ▼
  CodeBERT Classifier (reviewer_model.py)
  - Tokenizes each chunk (max 512 tokens)
  - Extracts [CLS] embedding (768-d)
  - Classification head → category + confidence
         │
         ▼
  Llama 3.2 Feedback Generator (Ollama)
  - Category-specific prompt construction
  - Local inference via Ollama API
  - Returns 1-2 sentence review comment
         │
         ▼
  FastAPI Backend → React Frontend
```

---

## 2. Model Training

### 2.1 Dataset

The base dataset is **CRAVE (Code Review Automated Validation and Evaluation)** — a public dataset of real GitHub pull request diffs with human reviewer labels. It contains 1,174 PR diff pairs (one approved, one requesting changes) from 50+ open source repositories across Python, TypeScript, Rust, C++, and other languages.

Raw label distribution after initial processing:

| Category | Count |
|---|---|
| ok | 454 |
| bug | 289 |
| style | 98 |
| security | 40 |
| performance | 27 |

### 2.2 Labelling Pipeline

The original CRAVE dataset only has binary labels (APPROVE / REQUEST_CHANGES) — it does not natively provide the 5-category classification our model needs. Two labelling approaches were attempted:

**Attempt 1 — Keyword matching (failed)**
The first approach used regex patterns on reviewer explanation text to derive labels. For example, if the explanation contained the word "vulnerability" it was tagged as "security". This produced near-random training signal — validation accuracy was only **31%** — because the keywords in reviewer explanations did not correlate with actual code-level patterns visible in the diffs.

**Attempt 2 — LLM-judged labels (used for training)**
Each diff was re-labelled using Gemini as an independent judge. The model was given each raw diff and asked to classify it into one of the 5 categories based on the code itself, not the reviewer's explanation. This produced significantly more consistent labels with 35.4% disagreement from the keyword-based labels — confirming the two approaches were doing fundamentally different things.

The Gemini-labelled dataset was used for all fine-tuning.

### 2.3 Fine-tuning CodeBERT

**Model architecture:**
```
microsoft/codebert-base (pretrained, 125M parameters)
    └── [CLS] token embedding (768-d)
        └── Dropout(0.1)
            └── Linear(768 → 5)
                └── Softmax → category probabilities
```

Only the classification head (Linear layer) was trained from scratch. The CodeBERT encoder weights were fine-tuned end-to-end with a low learning rate to preserve pretrained representations.

**Training setup:**

| Hyperparameter | Value |
|---|---|
| Base model | microsoft/codebert-base |
| Max sequence length | 512 tokens |
| Batch size | 8 |
| Epochs | 8 |
| Learning rate | 1e-5 |
| Optimizer | AdamW |
| Loss function | CrossEntropyLoss with inverse-frequency class weights |
| Validation strategy | Stratified 5-Fold Cross-Validation |

**Class weighting:**
The dataset is heavily imbalanced (ok=415 vs security=24). Without correction, the model would simply learn to predict "ok" or "bug" for everything and still achieve deceptively high accuracy. Inverse-frequency weights were applied to the loss function so the model is penalised more heavily for misclassifying minority classes:

```
weight[c] = total_samples / (num_classes × count[c])
```

**Stratified K-Fold:**
A single train/val/test split was avoided because minority classes (security: 24, performance: 39) are too small to be reliably represented in a fixed held-out set. Stratified 5-Fold ensures every class appears proportionally in every fold, and every example is used for both training and validation exactly once across the 5 folds — giving a more reliable estimate of true generalisation performance.

### 2.4 Results

Per-class report aggregated across all 5 folds:

```
              precision    recall  f1-score   support

         bug       0.82      0.85      0.84       324
       style       0.89      0.75      0.82       106
 performance       0.80      0.82      0.81        39
    security       0.89      0.67      0.76        24
          ok       0.82      0.85      0.83       415

    accuracy                           0.83       908
   macro avg       0.85      0.79      0.81       908
weighted avg       0.83      0.83      0.83       908
```

**83% accuracy, 0.81 macro F1** across 5 categories with genuine class imbalance. Notably, the security class achieved 0.89 precision despite only 24 training examples — suggesting CodeBERT's pretrained code representations encode meaningful security-related patterns that the classification head successfully learned to surface.

---

## 3. Backend — FastAPI

The backend (`backend/main.py`) exposes the full pipeline as a REST API using FastAPI with async request handling.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns model load status |
| POST | `/review/pr` | Reviews all files changed in a GitHub PR |
| POST | `/review/code` | Reviews raw pasted code |

**Key design decisions:**

**Models load once at startup, not per-request.** CodeBERT takes several seconds to initialise — loading it per-request would make the API unusable. FastAPI's `@app.on_event("startup")` hook loads both models into memory when the server starts, and they stay resident for the lifetime of the process.

**Blocking inference runs in a thread pool.** PyTorch inference is CPU/GPU-bound and blocks the Python event loop. Running it directly in an async endpoint would freeze all other requests while one review is in progress. `asyncio.to_thread()` offloads the blocking work to a thread pool, keeping the FastAPI event loop free.

**Structured Pydantic schemas.** Every request and response is validated against Pydantic models. This gives automatic type checking, clear API documentation at `/docs`, and prevents malformed data from reaching the model.

**CORS configured for local dev.** The frontend runs on port 5173 and the backend on port 8000 — CORS middleware allows cross-origin requests from both Vite's dev server and a standard React dev server.

**GitHub integration:**
The `github_fetcher.py` module calls the GitHub REST API (`/repos/{owner}/{repo}/pulls/{pr_number}/files`) to fetch the list of changed files and their raw diffs. Each file's patch is split into 30-line chunks — the maximum that fits cleanly within CodeBERT's 512-token limit after tokenization. Language is inferred from the file extension.

---

## 4. Frontend — React

The frontend (`frontend/src/`) is a single-page React app built with Vite.

**Layout:**
The app uses a two-column grid — a sidebar for input and a main panel for results. No routing is needed since there is only one view.

**Input modes:**
- **PR URL mode** — user pastes a GitHub PR URL, the frontend POSTs to `/review/pr`
- **Paste Code mode** — user pastes raw code directly, the frontend POSTs to `/review/code`

**Loading states:**
Since reviews can take 30–60+ seconds (Llama runs locally on CPU), the loading state cycles through descriptive stage labels ("Fetching source…", "Running CodeBERT…", "Generating feedback…") to give the user feedback that something is happening.

**Results rendering:**
Each changed file gets a `FileBlock` showing filename, language, and line additions/deletions. Within each file, actionable findings (non-"ok" classifications) are rendered as `Finding` cards with:
- A colour-coded category stamp (red=bug, violet=security, amber=performance, blue=style)
- A confidence bar showing the model's softmax probability for the predicted class
- The Llama-generated review comment
- The raw code chunk that triggered the finding

**Category colour system:**

| Category | Color |
|---|---|
| Bug | `#F85149` (red) |
| Security | `#A371F7` (violet) |
| Performance | `#D29922` (amber) |
| Style | `#58A6FF` (blue) |
| Clean | `#3FB950` (green) |

---

## 5. Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed with `llama3.2:1b` pulled
- A GitHub personal access token (classic, `repo` scope)

### Setup

**1. Clone and install Python dependencies:**
```bash
git clone https://github.com/yourusername/ai-code-reviewer.git
cd ai-code-reviewer
pip install -r requirements.txt
```

**2. Create a `.env` file in the root:**
```
GITHUB_TOKEN=your_github_token_here
```

**3. Download the fine-tuned model weights:**
The `.pt` file is not included in the repo due to size. Either:
- Run `fine_tune_codebert.py` to train your own (requires the Gemini-labelled dataset)
- Or the classifier will fall back to an untrained head with a warning

**4. Start Ollama:**
```bash
ollama pull llama3.2:1b
ollama serve
```

**5. Start the FastAPI backend:**
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

**6. Start the React frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## 6. Project Structure

```
.
├── reviewer.py                  # CLI entry point
├── reviewer_model.py            # CodeBERT classifier + Ollama feedback generator
├── github_fetcher.py            # GitHub API integration + diff chunking
├── fine_tune_codebert.py        # Stratified K-Fold fine-tuning script
├── build_training_data.py       # CRAVE dataset loader + keyword labelling
├── relabel_with_gemini.py       # LLM-judged relabelling pipeline
├── requirements.txt
├── .gitignore
│
├── backend/
│   └── main.py                  # FastAPI app — /review/pr and /review/code endpoints
│
└── frontend/
    ├── index.html
    ├── package.json
    └── src/
        ├── main.jsx
        ├── App.jsx              # Main React component
        └── App.css              # Styling
```

---

## License

MIT
