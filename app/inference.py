"""Loads the fine-tuned adapter and runs a single compliance analysis.

torch and transformers are imported lazily inside `load()`. That keeps the web
UI bootable on a machine where the ML stack is missing or the model has not
been trained yet — it shows a clear message instead of failing on import.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from shared.prompt_format import CATEGORIES, build_messages  # noqa: E402

ADAPTER_DIR = os.path.join(REPO_ROOT, "models", "financial-compliance-qwen", "final")
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Practical 1 analyses one snippet, not the whole document. Chunking and
# retrieval across a full filing arrive with the Hybrid RAG work.
SNIPPET_CHARS = 1200
MAX_NEW_TOKENS = 200


class ModelUnavailable(Exception):
    """The model cannot be served. The message is shown to the user."""


def adapter_present() -> bool:
    return os.path.isdir(ADAPTER_DIR) and any(
        os.path.exists(os.path.join(ADAPTER_DIR, f))
        for f in ("adapter_config.json", "adapter_model.safetensors")
    )


def make_snippet(text: str) -> str:
    """First SNIPPET_CHARS of normalised text, cut on a sentence if possible."""
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) <= SNIPPET_CHARS:
        return cleaned
    window = cleaned[:SNIPPET_CHARS]
    cut = max(window.rfind(". "), window.rfind(".\n"))
    return window[: cut + 1] if cut > SNIPPET_CHARS // 2 else window


def parse_model_output(raw: str) -> dict:
    """Pull the JSON object out of the model's reply, tolerating stray text.

    A 0.5B model will sometimes emit prose around the JSON or truncate it.
    Rather than 500 on the user, fall back to showing the raw text.
    """
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                category = str(parsed.get("category", "unknown"))
                return {
                    "issue_found": bool(parsed.get("issue_found", False)),
                    "category": category,
                    "category_label": CATEGORIES.get(category, category.replace("_", " ").title()),
                    "severity": str(parsed.get("severity", "unknown")),
                    "explanation": str(parsed.get("explanation", "")).strip(),
                    "parsed": True,
                }
        except json.JSONDecodeError:
            pass
    return {
        "issue_found": None,
        "category": "unparsed",
        "category_label": "Unstructured reply",
        "severity": "unknown",
        "explanation": raw.strip() or "(the model returned nothing)",
        "parsed": False,
    }


class ComplianceModel:
    """Lazily-loaded singleton around the base model plus the LoRA adapter."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str | None:
        return self._device

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:      # another thread won the race
                return
            if not adapter_present():
                raise ModelUnavailable(
                    "No trained adapter found. Train the model on the training "
                    f"laptop, then copy its 'final' folder to {ADAPTER_DIR}."
                )
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
            except ImportError as exc:
                raise ModelUnavailable(
                    "The inference stack is not installed. Run: "
                    f"pip install -r requirements-app.txt  ({exc})"
                ) from exc

            # No bitsandbytes here: the app machine may have no NVIDIA GPU.
            if torch.cuda.is_available():
                device, dtype = "cuda", torch.float16
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device, dtype = "mps", torch.float16
            else:
                device, dtype = "cpu", torch.float32

            try:
                tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            try:
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID, torch_dtype=dtype)
                model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            except Exception as exc:
                raise ModelUnavailable(f"Could not load the model: {exc}") from exc

            model.to(device)
            model.eval()
            self._tokenizer, self._model, self._device = tokenizer, model, device

    def analyze(self, snippet: str) -> dict:
        self.load()
        import torch

        prompt = self._tokenizer.apply_chat_template(
            build_messages(snippet), tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,            # greedy: we want stable JSON
                pad_token_id=self._tokenizer.pad_token_id,
            )
        # Keep only the newly generated continuation.
        reply = self._tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_model_output(reply)


MODEL = ComplianceModel()
