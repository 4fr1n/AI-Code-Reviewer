

"""
reviewer_model.py
-----------------
Two-stage NLP pipeline:
  Stage 1 — CodeBERT: encodes code chunks into embeddings,
             then classifies each into a feedback category
             (bug / style / performance / security / ok)
  Stage 2 — T5:  takes the category + code chunk and generates
             a natural language review comment
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel
)


print(f"CUDA available: {torch.cuda.is_available()}")
# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES    = ["bug", "style", "performance", "security", "ok"]
CODEBERT_NAME = "microsoft/codebert-base"

MAX_CODE_LEN  = 512


DEVICE = (
    "cuda"  if torch.cuda.is_available() else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)


# ── CodeBERT Classifier ───────────────────────────────────────────────────────

class CodeBERTClassifier(nn.Module):
    """
    Wraps CodeBERT with a classification head.
    Architecture:
        CodeBERT [CLS] embedding (768-d)
        → Dropout(0.1)
        → Linear(768 → 5)
        → Softmax → category probabilities
    """
    def __init__(self, num_labels: int = 5):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(CODEBERT_NAME)
        self.dropout  = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits

    def get_embedding(self, input_ids, attention_mask) -> torch.Tensor:
        """Returns the raw [CLS] embedding — useful for similarity search later."""
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


# ── Rule-based heuristic labels (used as pseudo-labels since we have no
#    labelled training data on Day 1 — the classifier is fine-tunable later) ──

HEURISTIC_PATTERNS = {
    "bug": [
        "null", "none", "undefined", "except", "error", "fail",
        "index", "overflow", "divide", "zero", "nan", "inf",
        "off by", "wrong", "incorrect", "broken",
    ],
    "security": [
        "password", "secret", "token", "api_key", "eval(",
        "exec(", "shell", "subprocess", "sql", "inject",
        "sanitize", "escape", "hash", "encrypt", "auth",
        "hardcoded", "plaintext",
    ],
    "performance": [
        "for loop", "nested", "o(n", "sleep", "wait", "block",
        "global", "cache", "memory", "leak", "redundant",
        "duplicate", "inefficient", "slow", "recompute",
        "n+1", "query inside loop",
    ],
    "style": [
        "naming", "comment", "docstring", "magic number",
        "long line", "unused", "import", "whitespace",
        "camel", "snake", "consistent", "readable",
    ],
}


def heuristic_label(code_chunk: str) -> str:
    """
    Simple pattern-based labeller used as a fallback when the
    classifier has low confidence. Scans the chunk for known keywords.
    """
    code_lower = code_chunk.lower()
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, patterns in HEURISTIC_PATTERNS.items():
        for p in patterns:
            if p in code_lower:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "ok"


# ── Feedback templates (for T5 prompt construction) ───────────────────────────

PROMPTS = {
    "bug": (
        "Review this code and identify potential bugs, "
        "off-by-one errors, null pointer issues, or incorrect logic:\n\n"
    ),
    "security": (
        "Review this code for security vulnerabilities such as "
        "hardcoded credentials, injection risks, or insecure patterns:\n\n"
    ),
    "performance": (
        "Review this code for performance issues such as "
        "inefficient loops, redundant computations, or memory problems:\n\n"
    ),
    "style": (
        "Review this code for style and readability issues such as "
        "naming conventions, missing docstrings, or magic numbers:\n\n"
    ),
    "ok": (
        "Briefly confirm this code looks clean and note anything worth keeping:\n\n"
    ),
}


# ── Main Reviewer Pipeline ────────────────────────────────────────────────────

class CodeReviewer:
    def __init__(self, weights_path: str = "codebert_classifier_finetuned.pt"):
        print(f"  Loading models on {DEVICE.upper()} …")

        # CodeBERT tokenizer + classifier
        print("  Loading CodeBERT …")
        self.cb_tokenizer  = AutoTokenizer.from_pretrained(CODEBERT_NAME)
        self.cb_classifier = CodeBERTClassifier(num_labels=len(CATEGORIES))

        # Load fine-tuned classification head weights (trained via
        # fine_tune_codebert.py on Gemini-labelled CRAVE data, ~70% val accuracy)
        import os
        if os.path.exists(weights_path):
            print(f"  Loading fine-tuned weights from {weights_path} …")
            state_dict = torch.load(weights_path, map_location=DEVICE)
            self.cb_classifier.load_state_dict(state_dict)
            print("  ✓ Fine-tuned weights loaded")
        else:
            print(f"  ⚠ WARNING: {weights_path} not found — using untrained "
                  f"classification head. Classifications will be near-random. "
                  f"Run fine_tune_codebert.py first or place the .pt file in "
                  f"this directory.")

        self.cb_classifier.to(DEVICE)
        self.cb_classifier.eval()

        print("  Models ready.\n")

    def _encode_chunk(self, code: str):
        """Tokenize a code chunk for CodeBERT."""
        return self.cb_tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CODE_LEN,
            padding="max_length",
        )

    def classify(self, code_chunk: str) -> tuple[str, float]:
        """
        Runs CodeBERT classification using the fine-tuned model.
        Returns (category, confidence).
        """
        encoded = self._encode_chunk(code_chunk)
        input_ids      = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = self.cb_classifier(input_ids, attention_mask)

        probs      = torch.softmax(logits, dim=-1)[0]
        confidence = probs.max().item()
        pred_idx   = probs.argmax().item()
        category   = CATEGORIES[pred_idx]

        return category, confidence

    def get_embedding(self, code_chunk: str) -> torch.Tensor:
        """Returns CodeBERT [CLS] embedding for a chunk (for similarity search)."""
        encoded = self._encode_chunk(code_chunk)
        input_ids      = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        return self.cb_classifier.get_embedding(input_ids, attention_mask)

    def generate_feedback(self, code_chunk: str, category: str) -> str:
        import requests

        prompt = f"""You are a code reviewer. Analyse this code and give ONE specific,
                    concise review comment focused on {category} issues.
                    Be direct and actionable. Max 2 sentences.

        Code:
        {code_chunk}

        Review comment:"""

        try:
            response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
            return response.json()["response"].strip()
        except Exception as e:
            return f"(feedback unavailable: {e})"

    def review_chunk(self, code_chunk: str) -> dict:
        """
        Full pipeline for one chunk:
          1. Classify with CodeBERT
          2. Generate feedback with T5
        Returns a result dict.
        """
        category, confidence = self.classify(code_chunk)
        feedback = self.generate_feedback(code_chunk, category)

        return {
            "category":   category,
            "confidence": confidence,
            "feedback":   feedback,
            "chunk":      code_chunk,
        }

    def review_file(self, file_info: dict) -> list[dict]:
        """
        Reviews all chunks in a file. Skips 'ok' chunks to reduce noise.
        Returns only actionable findings.
        """
        results = []
        for i, chunk in enumerate(file_info["chunks"]):
            result = self.review_chunk(chunk)
            result["chunk_index"] = i
            result["filename"]    = file_info["filename"]
            result["language"]    = file_info["language"]
            results.append(result)
        return results
