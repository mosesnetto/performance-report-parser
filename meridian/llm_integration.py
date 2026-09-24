"""
Phase 4 — LLM Integration (free model via OpenRouter).
Read-only query interface using structured query tools (app/query_tools).
Does NOT expose API keys in HTML/logs.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

# Load from environment; never expose to frontend
# Supports single key or multiple keys separated by commas
API_KEYS_RAW = os.getenv("OPENROUTER_API_KEYS", os.getenv("OPENROUTER_API_KEY", ""))
API_KEYS = [k.strip() for k in API_KEYS_RAW.split(",") if k.strip()] if API_KEYS_RAW else []
MODEL = os.getenv("AI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
FALLBACK = os.getenv("AI_FALLBACK_MODEL", "openrouter/free")

# List of free/openrouter models to try sequentially if previous fails
FREE_MODELS = [
    MODEL,
    FALLBACK,
    "openrouter/free",
    "qwen/qwen-2.5-72b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-3-27b-it:free",
]

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _call_model(prompt: str, model: str, key_index: int = 0) -> Optional[str]:
    if not API_KEYS:
        return f"[LLM unavailable: OPENROUTER_API_KEY(S) not set. Model: {model}]"
    api_key = API_KEYS[key_index % len(API_KEYS)]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    try:
        resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return "[LLM returned empty response]"
    except Exception as exc:
        return f"[LLM error: {type(exc).__name__}: {str(exc)[:200]}]"


def query_with_llm(question: str) -> Dict[str, Any]:
    """Main analytics query interface.
    Uses deterministic query tools first; LLM provides interpretation/explanation."""
    from meridian.analytics import get_analytics_registry
    from meridian.query_tools import (
        search_fields,
    )

    # Build structured evidence first (deterministic)
    registry = get_analytics_registry()
    evidence = {
        "question": question,
        "model": MODEL,
        "fields_registered": len(registry.get("fields", [])),
        "reports_present": registry.get("reports_present", []),
    }

    # Try to identify relevant field from question
    matched_fields = search_fields(question)
    evidence["matched_fields"] = [f.get("canonical_field") for f in matched_fields[:5]]

    # Call LLM for reasoning (without exposing key)
    # Include ALL extracted tabular data in evidence
    all_fields_summary = []
    for f in registry.get("fields", [])[:20]:  # Show all registered fields briefly
        field_name = f.get("canonical_field", f.get("canonical_field", ""))
        status = f.get("status", "")
        reports = [r.get("pdf_name", "") for r in f.get("source_reports", []) if r.get("pdf_name")]
        all_fields_summary.append(f"{field_name} ({status}) in {len(reports)} reports")
    
    prompt = f"""You are an analytics assistant for performance report data.
Use ALL available structured extracted tabular data below to answer.
Answer concisely with evidence and cite actual reports.

User question: {question}

FULL EXTRACTED DATA SUMMARY (use all):
- Registered fields: {len(registry.get('fields', []))}
- Reports available: {registry.get('reports_present', [])}
- All fields ({len(all_fields_summary)} shown briefly): {', '.join(all_fields_summary)}
- Matched fields for query: {evidence['matched_fields']}

Instructions: Inspect the full tabular evidence. Compare values across reports. Cite report filenames, regions, values, and units. Do not invent values."""

# Try all available free models + API keys quickly; find fastest available
    tried_combos = set()
    # Prioritize faster/free endpoints first (reordered for speed)
    speed_order = [
        "openrouter/free",
        MODEL,
        FALLBACK,
        "qwen/qwen-2.5-72b-instruct:free",
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.1-8b-instruct:free",
    ]
    unique_models = []
    for m in speed_order + FREE_MODELS:
        if m not in unique_models:
            unique_models.append(m)
    key_index = 0
    for model in unique_models:
        combo_key = (model, key_index)
        if combo_key in tried_combos:
            key_index += 1
            combo_key = (model, key_index)
        tried_combos.add(combo_key)
        answer_text = _call_model(prompt, model, key_index)
        # Rotate to next key immediately (faster cycle)
        key_index = (key_index + 1) % max(len(API_KEYS), 1)
        if answer_text and "LLM unavailable" not in answer_text and "LLM error" not in answer_text:
            break
        # Minimal delay for rate-limit recovery (0.1s)
        import time
        time.sleep(0.1)
    else:
        # All models/keys exhausted; use structured fallback
        from meridian.query_tools import search_fields
        matched = search_fields(question)[:3]
        evidence_str = ", ".join([f.get("canonical_field") for f in matched]) if matched else "none"
        answer_text = f"[LLM unavailable after retries across all free models/keys. Using structured evidence.] Question: {question}. Relevant fields: {evidence_str}. Reports available: see analytics store."

    return {
        "ok": True,
        "question": question,
        "answer": answer_text,
        "evidence": evidence,
        "source_model": MODEL,
        "fallback_attempted": (answer_text != _call_model(prompt, MODEL) if answer_text else False),
    }


def explain_trend(field_name: str, explanation: str = "trend") -> Dict[str, Any]:
    """Generate explanation for a specific field trend."""
    from meridian.query_tools import aggregate_field
    stats = aggregate_field(field_name, operation="mean")
    prompt = f"""Explain the trend/status for parameter '{field_name}'.
Statistics: {stats}
Keep explanation concise. Cite actual reports from source data."""
    return {
        "ok": True,
        "field": field_name,
        "explanation": _call_model(prompt, MODEL) or "No explanation available.",
        "statistics": stats,
    }
