"""
Abnormal Detection Node
========================
Flags numeric results outside reference range → LOW / HIGH.
Triages qualitative POSITIVE results via AI to decide clinical significance.

File: app/services/abnormal_node.py

ROOT CAUSE of BioLLM always falling back to Gemini
────────────────────────────────────────────────────
The HF router sends OpenBioLLM requests to featherless-ai which exposes
/v1/completions (the OLD text-completion endpoint), NOT /v1/chat/completions.

When you call client.chat_completion(), huggingface_hub internally hits
/v1/chat/completions — but the router silently re-routes it to /v1/completions
and returns a response whose shape differs from what chat_completion() expects.
Specifically, choices[0].message is None or missing, so .content returns ""
or raises AttributeError, _parse_triage_response gets an empty string, returns
None, and the fallback fires — even though HTTP was 200.

The log proof:
    POST https://router.huggingface.co/featherless-ai/v1/completions → 200 OK
    WARNING | BioLLM triage inconclusive — falling back to Gemini

FIX: use client.text_generation() which maps directly to /v1/completions —
exactly what featherless-ai serves. The response is a plain string, not a
ChatCompletionOutput object, so there is no shape mismatch.

All other fixes retained:
  • Full context block injected into every prompt (stateless model, zero memory)
  • Robust response parsing (substring search, paraphrase acceptance)
  • stop_sequences=["\n", "."] so model stops after its one-word answer
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.services.llm_clients import call_gemini, get_biollm_client

log = logging.getLogger(__name__)

_POSITIVE_VALUES = {"positive", "reactive", "detected", "present", "abnormal"}
_NEGATIVE_VALUES = {"negative", "non-reactive", "not detected", "absent", "normal"}


# ─────────────────────────────────────────────────────────────
# Range helpers
# ─────────────────────────────────────────────────────────────

def _parse_range(range_str: str) -> Optional[Tuple[float, float]]:
    range_str = str(range_str).strip()
    m = re.match(r"^([\d.]+)\s*[-–]\s*([\d.]+)$", range_str)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^<\s*([\d.]+)$", range_str)
    if m:
        return 0.0, float(m.group(1))
    m = re.match(r"^>\s*([\d.]+)$", range_str)
    if m:
        return float(m.group(1)), float("inf")
    return None


def _numeric_status(value_str: str, range_str: str) -> Optional[str]:
    cleaned = re.sub(r"[^\d.\-]", "", str(value_str))
    try:
        value = float(cleaned)
    except ValueError:
        return None
    bounds = _parse_range(range_str)
    if bounds is None:
        return None
    low, high = bounds
    if value < low:
        return "LOW"
    if value > high:
        return "HIGH"
    return "NORMAL"


# ─────────────────────────────────────────────────────────────
# Context builder
# The model is completely stateless — inject everything it needs
# into every single call.
# ─────────────────────────────────────────────────────────────

def _build_context_block(state: Dict, focus_test: str, focus_value: str) -> str:
    report    = state.get("structured_report", {})
    patient   = report.get("patient_info", {})
    all_tests = report.get("test_results", [])

    lines: List[str] = [
        f"Patient: {patient.get('age', 'unknown')} years old, {patient.get('gender', 'unknown')}",
        "\nAll test results in this report:",
    ]
    for t in all_tests:
        lines.append(
            f"  {t.get('test', '')}: {t.get('value', '')}  (ref: {t.get('range', 'N/A')})"
        )
    lines.append(f"\nTest under review: {focus_test} = {focus_value}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Prompt — plain completion string
# Matches /v1/completions which is what featherless-ai serves.
# ─────────────────────────────────────────────────────────────

_TRIAGE_TEMPLATE = """\
You are a clinical pathologist reviewing a lab report.

{context_block}

Question: Is the result for "{test_name}" clinically significant and worth flagging to the patient?

Rules:
- Reply with EXACTLY one word.
- Use SIGNIFICANT if the result needs medical attention.
- Use NOT_SIGNIFICANT if the result is clinically unimportant.
- Do not add any other text, punctuation, or explanation.

Answer:"""


# ─────────────────────────────────────────────────────────────
# Response parser — robust against preamble prose
# ─────────────────────────────────────────────────────────────

def _parse_triage_response(raw: str) -> Optional[bool]:
    """
    Searches the full response for keywords.
    Checks NOT_SIGNIFICANT before SIGNIFICANT to avoid substring collision.
    Returns None only when genuinely ambiguous → triggers fallback.
    """
    upper = raw.strip().upper()
    log.info("  [BioLLM] Full raw response: %r", raw.strip()[:200])

    if "NOT_SIGNIFICANT" in upper or "NOT SIGNIFICANT" in upper:
        return False
    if "SIGNIFICANT" in upper:
        return True
    if any(x in upper for x in ["NOT CLINICALLY", "CLINICALLY INSIGNIFICANT", "NO CLINICAL"]):
        return False
    if any(x in upper for x in ["CLINICALLY SIGNIFICANT", "NEEDS ATTENTION", "SHOULD BE FLAGGED"]):
        return True

    log.warning("  [BioLLM] Response not parseable: %r", raw[:200])
    return None


# ─────────────────────────────────────────────────────────────
# BioLLM call — text_generation() → /v1/completions
#
# KEY FIX: switched from client.chat_completion() to
# client.text_generation() because the HF router proxies
# OpenBioLLM via featherless-ai which only serves /v1/completions.
# chat_completion() sends to /v1/chat/completions; the router
# re-routes it to /v1/completions anyway but wraps the reply in
# a ChatCompletionOutput whose .choices[0].message.content is
# empty — causing our parser to return None every time.
# text_generation() returns a plain string directly. No shape
# mismatch, no silent empty result.
# ─────────────────────────────────────────────────────────────

def _triage_via_biollm(
    test_name: str,
    result_value: str,
    context_block: str,
) -> Optional[bool]:
    prompt = _TRIAGE_TEMPLATE.format(
        context_block=context_block,
        test_name=test_name,
    )
    try:
        client = get_biollm_client()
        log.info("  [BioLLM] Sending triage request for '%s'...", test_name)

        raw: str = client.text_generation(
            prompt,
            max_new_tokens=16,
            temperature=0.05,
            stop_sequences=["\n", "\n\n", ".", "Question:"],
            do_sample=False,
        )

        log.info("  [BioLLM] Raw response for '%s': %r", test_name, raw[:200])
        return _parse_triage_response(raw)

    except Exception as exc:
        exc_str = str(exc)
        if "503" in exc_str or "loading" in exc_str.lower():
            log.warning("  [BioLLM] Model loading (cold start 503) for '%s': %s", test_name, exc)
        elif "401" in exc_str or "403" in exc_str or "authorization" in exc_str.lower():
            log.error("  [BioLLM] Auth error for '%s' — check HF_TOKEN: %s", test_name, exc)
        else:
            log.warning("  [BioLLM] Call failed for '%s': %s", test_name, exc)
        return None


# ─────────────────────────────────────────────────────────────
# Gemini fallback
# ─────────────────────────────────────────────────────────────

_GEMINI_TRIAGE_TEMPLATE = """\
You are a clinical pathologist reviewing a lab report.

{context_block}

Is the result for "{test_name}" clinically significant?
Reply with EXACTLY one word: SIGNIFICANT or NOT_SIGNIFICANT.
"""


def _triage_via_gemini(
    test_name: str,
    result_value: str,
    context_block: str,
) -> bool:
    prompt = _GEMINI_TRIAGE_TEMPLATE.format(
        context_block=context_block,
        test_name=test_name,
    )
    try:
        answer = call_gemini(prompt).strip()
        log.info("  [Gemini fallback] Response for '%s': %r", test_name, answer[:120])
        result = _parse_triage_response(answer)
        if result is None:
            log.warning(
                "  [Gemini fallback] Still unparseable for '%s' — defaulting to SIGNIFICANT.",
                test_name
            )
            return True
        return result
    except Exception as exc:
        log.error("  [Gemini fallback] Also failed for '%s': %s", test_name, exc)
        return True


# ─────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────

def _is_qualitative_significant(
    test_name: str,
    result_value: str,
    context_block: str,
) -> bool:
    log.info("  Triaging qualitative: %s = %s", test_name, result_value)

    result = _triage_via_biollm(test_name, result_value, context_block)

    if result is not None:
        log.info(
            "  [BioLLM] Decision for '%s': %s",
            test_name, "SIGNIFICANT" if result else "NOT_SIGNIFICANT"
        )
        return result

    log.warning("  BioLLM inconclusive for '%s' — falling back to Gemini.", test_name)
    result = _triage_via_gemini(test_name, result_value, context_block)
    log.info(
        "  [Gemini fallback] Decision for '%s': %s",
        test_name, "SIGNIFICANT" if result else "NOT_SIGNIFICANT"
    )
    return result


# ─────────────────────────────────────────────────────────────
# Node entry point
# ─────────────────────────────────────────────────────────────

def abnormal_node(state: Dict) -> Dict:
    log.info("=== abnormal_node: START ===")

    structured    = state.get("structured_report", {})
    test_results: List[Dict] = structured.get("test_results", [])
    abnormal_values: Dict    = {}

    log.info("  Total test results to scan: %d", len(test_results))

    for entry in test_results:
        test_name  = str(entry.get("test",  "")).strip()
        raw_value  = str(entry.get("value", "")).strip()
        range_str  = str(entry.get("range", "")).strip()

        if not test_name or not raw_value:
            continue

        value_lower = raw_value.lower()

        # ── Numeric ───────────────────────────────────────────
        if range_str and value_lower not in (_POSITIVE_VALUES | _NEGATIVE_VALUES):
            status = _numeric_status(raw_value, range_str)
            if status in ("LOW", "HIGH"):
                try:
                    numeric = float(re.sub(r"[^\d.\-]", "", raw_value))
                except ValueError:
                    numeric = raw_value
                abnormal_values[test_name] = {
                    "value": numeric, "range": range_str, "status": status
                }
                log.info("  Numeric flagged: %s = %s [%s]", test_name, numeric, status)
            else:
                log.debug("  Numeric normal/unparseable: %s", test_name)
            continue

        # ── Qualitative ───────────────────────────────────────
        if value_lower in _POSITIVE_VALUES:
            context_block = _build_context_block(state, test_name, raw_value)
            if _is_qualitative_significant(test_name, raw_value, context_block):
                abnormal_values[test_name] = {
                    "value": raw_value, "range": "", "status": raw_value.upper()
                }
                log.info("  Qualitative flagged: %s = %s", test_name, raw_value)
            else:
                log.info("  Qualitative NOT significant: %s", test_name)

        elif value_lower in _NEGATIVE_VALUES:
            log.debug("  Qualitative NEGATIVE — skipped: %s", test_name)

    log.info("=== abnormal_node: DONE — %d abnormal(s) ===", len(abnormal_values))
    for name, data in abnormal_values.items():
        log.info("  • %s: %s [%s]", name, data["value"], data["status"])

    return {**state, "abnormal_values": abnormal_values}