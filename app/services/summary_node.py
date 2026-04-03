"""
Summary Node
=============
Generates a structured AI clinical summary using BioLLM → Gemini → rule fallback.

File: app/services/summary_node.py

FIX: same root cause as abnormal_node — featherless-ai serves /v1/completions
so client.text_generation() must be used instead of client.chat_completion().
Also added stop_sequences to prevent the model rambling past the structured
output block, and bumped max_new_tokens to 512 to avoid truncated summaries.

PROMPT SAFETY UPDATE (v2):
- Strengthened NO DIAGNOSIS rule with explicit banned phrases
- Controlled interpretation language (possibility, not certainty)
- Prevented hallucinations (no unsupported conditions)
- Limited condition mentions to once (no repetition)
- SUMMARY is purely descriptive (no inference)
- INTERPRETATION is non-definitive
- RECOMMENDATION is consult-only (no treatment advice)
- Output length constrained for Twilio compatibility
- Each section output exactly once
- Hard stop after RECOMMENDATION
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.services.llm_clients import call_gemini, get_biollm_client

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────

def _build_context(state: Dict) -> str:
    patient_info    = state.get("structured_report", {}).get("patient_info", {})
    abnormal_values = state.get("abnormal_values", {})
    explanation     = state.get("explanation",     {})
    patterns        = state.get("patterns",        [])

    lines: List[str] = [
        f"Patient: {patient_info.get('name','Unknown')}, "
        f"{patient_info.get('age','Unknown')}, "
        f"{patient_info.get('gender','Unknown')}. "
        f"Report date: {patient_info.get('report_date','Unknown')}."
    ]

    if abnormal_values:
        lines.append("\nAbnormal Lab Values:")
        for test, data in abnormal_values.items():
            val    = data.get("value",  "N/A")
            rng    = data.get("range",  "N/A")
            status = data.get("status", "N/A")
            exp    = explanation.get(test, "")
            lines.append(f"  - {test}: {val} (ref: {rng}) [{status}]")
            if exp:
                lines.append(f"    Note: {exp}")
    else:
        lines.append("\nNo significant abnormal values detected.")

    if patterns:
        lines.append("\nDetected Clinical Patterns:")
        for p in patterns:
            lines.append(f"  - {p}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Prompt — plain completion string (matches /v1/completions)
# ─────────────────────────────────────────────────────────────
_PROMPT = """\
You are a clinical AI assistant helping patients understand their lab reports.

### PATIENT DATA:
{context}

### TASK:
Generate a structured medical report explanation using the format below.
Output each section EXACTLY ONCE. Do NOT repeat any section heading or content.
STOP immediately after RECOMMENDATION. Do NOT add any text after it.

---

SUMMARY:
<Describe ONLY the abnormal values observed. State the test name and whether it is high or low.
Do NOT infer, suggest, or mention any disease or condition here.
Example: "Elevated glucose and low platelet count were observed."
Keep to 2 sentences maximum.>

INTERPRETATION:
<Connect the findings to possible general meanings using cautious language.
You MAY mention a condition name here — but AT MOST ONCE, and only if strongly supported by the data.
After first mention, refer to it as "this finding" or "these results".
Do NOT say "you have", "indicates", "confirms", or "diagnosed with".
ONLY use: "suggests", "may indicate", "is consistent with", "raises suspicion for".
Do NOT escalate to a confirmed disease. Keep to 2 sentences maximum.>

RISK_LEVEL:
<ONE of: Low / Moderate / High — based strictly on the data provided>

ALERTS:
<List each alert on a new line starting with "->".
Only include alerts directly supported by the lab values.
Do NOT invent alerts for conditions not present in the data.
Write "None" if there are no alerts.>

RECOMMENDATION:
<Advise the patient to consult a doctor or seek further evaluation.
Do NOT prescribe medication, diet, or lifestyle changes.
Do NOT give direct medical instructions.
Only suggest: consulting a physician, monitoring, or further tests.
Keep to 1-2 sentences.>

---

### ABSOLUTE BANNED PHRASES — NEVER USE THESE:
- "you have"
- "you are diagnosed"
- "indicates"
- "confirms"
- "diagnosed with"
- "you suffer from"
- "this proves"

### ONLY ALLOWED INTERPRETIVE PHRASES:
- "suggests"
- "may indicate"
- "is consistent with"
- "raises suspicion for"
- "could be associated with"

### HALLUCINATION PREVENTION:
- Do NOT mention any condition (e.g. Cushing's syndrome, lupus, cancer) unless the lab data explicitly supports it.
- Do NOT pattern-match aggressively. Stick only to what the numbers show.
- Do NOT add diseases based on vague associations.

### REPETITION RULES:
- Each condition or finding may be named AT MOST ONCE across the entire output.
- After the first mention, use "this condition", "these findings", or "this result".
- Do NOT repeat the same point in different sections.

### LENGTH RULES:
- Total output must be under 1000 characters.
- Use short sentences. No long explanations.
- Prefer clarity over completeness.

### OUTPUT:
SUMMARY:
"""


# ─────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────

# Phrases that indicate the model slipped into diagnosis tone
_BANNED_PHRASES = [
    r"\byou have\b",
    r"\byou are diagnosed\b",
    r"\bdiagnosed with\b",
    r"\bconfirms\b",
    r"\bindicates\b",
    r"\byou suffer from\b",
    r"\bthis proves\b",
]
_BANNED_RE = re.compile("|".join(_BANNED_PHRASES), re.IGNORECASE)


def _sanitise(text: str) -> str:
    """
    Post-generation safety net: replace banned diagnosis phrases with
    safe alternatives so the UI never shows overconfident language,
    even if the model partially ignores the prompt rules.
    """
    replacements = {
        r"\byou have\b":          "findings suggest",
        r"\byou are diagnosed\b": "results may indicate",
        r"\bdiagnosed with\b":    "consistent with possible",
        r"\bconfirms\b":          "is consistent with",
        r"\bindicates\b":         "suggests",
        r"\byou suffer from\b":   "findings may be associated with",
        r"\bthis proves\b":       "this raises suspicion for",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _parse_output(text: str) -> Dict:
    # Prepend the label we used as stop-prompt so the regex finds it
    if not text.lstrip().upper().startswith("SUMMARY"):
        text = "SUMMARY:\n" + text

    # ── Safety sanitisation (catches model slip-throughs) ────
    text = _sanitise(text)

    def _extract(label: str) -> str:
        m = re.search(
            rf"{label}\s*:\s*\n?(.*?)(?=\n[A-Z_]{{3,}}\s*:|$)",
            text, re.DOTALL | re.IGNORECASE
        )
        return m.group(1).strip() if m else ""

    risk_raw = _extract("RISK_LEVEL")
    risk = "Moderate"
    for level in ("High", "Moderate", "Low"):
        if level.lower() in risk_raw.lower():
            risk = level
            break

    alerts_raw = _extract("ALERTS")
    alerts: List[str] = []
    if alerts_raw.strip().lower() not in ("none", ""):
        for line in alerts_raw.splitlines():
            line = re.sub(r"^[-•*>]+\s*", "", line).strip()
            if line and line.lower() != "none":
                alerts.append(line)

    summary        = _extract("SUMMARY")
    interpretation = _extract("INTERPRETATION")
    recommendation = _extract("RECOMMENDATION")

    # ── Deduplication: remove repeated sentences across sections ──
    seen_sentences: set = set()

    def _deduplicate(block: str) -> str:
        out = []
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            key = sentence.strip().lower()
            if key and key not in seen_sentences:
                seen_sentences.add(key)
                out.append(sentence.strip())
        return " ".join(out)

    summary        = _deduplicate(summary)
    interpretation = _deduplicate(interpretation)
    recommendation = _deduplicate(recommendation)

    log.info("  [Parser] summary        : %r", summary[:80] if summary else "")
    log.info("  [Parser] interpretation : %r", interpretation[:80] if interpretation else "")
    log.info("  [Parser] risk           : %r", risk)
    log.info("  [Parser] alerts         : %s", alerts)
    log.info("  [Parser] recommendation : %r", recommendation[:80] if recommendation else "")

    # ── Warn if banned phrases survived sanitisation ─────────
    combined = f"{summary} {interpretation} {recommendation}"
    if _BANNED_RE.search(combined):
        log.warning("  [Parser] Banned phrase still present after sanitisation!")

    if not summary or not interpretation:
        log.warning("  [Parser] Missing required fields — parse failed.")
        return {}

    return {
        "summary":        summary,
        "interpretation": interpretation,
        "risk":           risk,
        "alerts":         alerts,
        "recommendation": recommendation or "Consult a physician for clinical correlation.",
    }


# ─────────────────────────────────────────────────────────────
# Rule-based fallback
# ─────────────────────────────────────────────────────────────

def _rule_summary(state: Dict) -> Dict:
    abnormal = state.get("abnormal_values", {})
    patterns = state.get("patterns", [])
    names    = list(abnormal.keys())
    count    = len(names)

    summary = (
        f"Abnormal values were observed in: {', '.join(names)}."
        if names else "No major abnormalities detected."
    )
    interpretation = (
        f"These results may be consistent with: {', '.join(patterns[:2])}." if patterns
        else "No significant clinical patterns identified."
    )
    risk = "High" if count >= 3 else ("Moderate" if count >= 1 else "Low")

    alerts: List[str] = []
    names_lower = [k.lower() for k in names]
    if any("platelet" in n for n in names_lower):
        alerts.append("Low platelet count — increased bleeding risk.")
    if any("wbc" in n or "leukocyte" in n for n in names_lower):
        alerts.append("Low white blood cell count — possible immune suppression.")
    if any("dengue" in n for n in names_lower):
        alerts.append("Dengue marker detected — monitor for haemorrhagic complications.")

    recommendation = (
        "Immediate medical consultation is strongly advised."
        if risk == "High" else
        "Consult a physician if symptoms persist or worsen."
    )
    return {
        "summary": summary, "interpretation": interpretation,
        "risk": risk, "alerts": alerts, "recommendation": recommendation,
    }


# ─────────────────────────────────────────────────────────────
# LLM callers
# ─────────────────────────────────────────────────────────────

def _call_biollm(context: str) -> Optional[Dict]:
    """
    FIX: use text_generation() — featherless-ai serves /v1/completions.
    The prompt ends with "SUMMARY:" so the model continues directly into
    the structured output. stop_sequences cuts it off after RECOMMENDATION.
    max_new_tokens=512 prevents truncated summaries.
    """
    try:
        client = get_biollm_client()
        prompt = _PROMPT.format(context=context)
        log.info("  [BioLLM] Sending summary request...")

        raw = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.2,
            stop_sequences=["### ", "RULES:", "You are"],
            do_sample=False,
        )

        log.info("  [BioLLM] Raw response (first 400 chars):\n%s", raw[:400])
        parsed = _parse_output(raw)
        if not parsed:
            log.warning("  [BioLLM] Parsing failed on response above.")
        return parsed if parsed else None

    except Exception as exc:
        exc_str = str(exc)
        if "503" in exc_str or "loading" in exc_str.lower():
            log.warning("  [BioLLM] Model loading (cold start): %s", exc)
        elif "401" in exc_str or "403" in exc_str:
            log.error("  [BioLLM] Auth error — check HF_TOKEN: %s", exc)
        else:
            log.warning("  [BioLLM] Call failed: %s", exc)
        return None


def _call_gemini(context: str) -> Optional[Dict]:
    try:
        prompt = _PROMPT.format(context=context)
        log.info("  [Gemini] Sending summary request...")
        raw    = call_gemini(prompt)
        log.info("  [Gemini] Raw response (first 400 chars):\n%s", raw[:400])
        parsed = _parse_output(raw)
        if not parsed:
            log.warning("  [Gemini] Parsing failed on response above.")
        return parsed if parsed else None
    except Exception as exc:
        log.error("  [Gemini] Call failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────
# Node entry point
# ─────────────────────────────────────────────────────────────

def summary_node(state: Dict) -> Dict:
    log.info("=== summary_node: START ===")
    context = _build_context(state)
    log.info("  Context built (%d chars).", len(context))
    log.debug("  Full context:\n%s", context)

    # ── Try BioLLM ────────────────────────────────────────────
    final_output = _call_biollm(context)
    engine       = "biollm"

    # ── Try Gemini ────────────────────────────────────────────
    if not final_output:
        log.warning("  BioLLM summary failed — falling back to Gemini.")
        final_output = _call_gemini(context)
        engine       = "gemini"

    # ── Rule-based last resort ────────────────────────────────
    if not final_output:
        log.error("  Gemini also failed — using rule-based fallback.")
        final_output = _rule_summary(state)
        engine       = "rules_only"

    final_output["summary_engine"] = engine
    log.info("=== summary_node: DONE — engine=%s, risk=%s ===",
             engine, final_output.get("risk"))
    return {**state, "final_output": final_output}