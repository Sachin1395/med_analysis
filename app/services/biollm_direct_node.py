"""
BioLLM Direct Pattern Node
============================
Compresses abnormal signals → single BioLLM call → 2-3 clean condition names.

File: app/services/biollm_direct_node.py
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.services.llm_clients import call_gemini, get_biollm_client

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Signal compression map
# ─────────────────────────────────────────────────────────────
_SIGNAL_MAP: Dict[tuple, str] = {
    ("wbc count",                    "LOW"):      "Leukopenia (low WBC)",
    ("wbc count",                    "HIGH"):     "Leukocytosis (high WBC)",
    ("hemoglobin",                   "LOW"):      "Anemia (low Hb)",
    ("hemoglobin",                   "HIGH"):     "Polycythemia (high Hb)",
    ("rbc count",                    "LOW"):      "Erythropenia (low RBC)",
    ("rbc count",                    "HIGH"):     "Erythrocytosis (high RBC)",
    ("platelet count",               "LOW"):      "Thrombocytopenia (low platelets)",
    ("platelet count",               "HIGH"):     "Thrombocytosis (high platelets)",
    ("packed cell volume [pcv]",     "HIGH"):     "Elevated PCV / polycythemia",
    ("packed cell volume [pcv]",     "LOW"):      "Reduced PCV / anemia",
    ("eosinophils",                  "LOW"):      "Eosinopenia",
    ("eosinophils",                  "HIGH"):     "Eosinophilia",
    ("esr",                          "HIGH"):     "Elevated ESR (systemic inflammation)",
    ("crp",                          "HIGH"):     "Elevated CRP (acute inflammation)",
    ("dengue antigen (ns1)- serum",  "POSITIVE"): "Dengue NS1 Antigen POSITIVE",
    ("dengue-igm antibodies- serum", "POSITIVE"): "Dengue IgM POSITIVE (acute infection)",
    ("dengue-igg antibodies- serum", "POSITIVE"): "Dengue IgG POSITIVE (past infection)",
}

_GENERIC = {
    "LOW":      "{test} is LOW",
    "HIGH":     "{test} is HIGH",
    "POSITIVE": "{test} is POSITIVE",
    "REACTIVE": "{test} is REACTIVE",
}


def _compress_signals(abnormal_values: Dict) -> str:
    lines: List[str] = []
    for raw_name, data in abnormal_values.items():
        status  = str(data.get("status", "")).upper()
        key     = (raw_name.lower().strip(), status)
        signal  = _SIGNAL_MAP.get(key)
        if signal:
            lines.append(f"- {signal}")
        elif status in _GENERIC:
            lines.append(f"- {_GENERIC[status].format(test=raw_name)}")
    return "\n".join(lines) if lines else "- No specific signals available"


# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────
_PROMPT = """\
You are a clinical medical expert AI.

A patient's lab report shows these abnormal findings:
{signals}

List 2 to 3 possible medical conditions that could explain these findings.

STRICT RULES:
- Output ONLY a numbered list of condition names.
- No explanations, no sentences, no notes, no warnings, no preamble.
- Start directly with "1.".
- Use standard medical terminology.

Example:
1. Dengue fever
2. Viral thrombocytopenia
3. Acute febrile illness

Answer:"""

_NOISE_TOKENS = {
    "note", "consult", "professional", "disclaimer", "output",
    "instruction", "warning", "caution", "please", "however",
    "important", "always", "seek", "advice",
}



def _parse_patterns(text: str, max_items: int = 3) -> List[str]:
    results: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(\d+[.)]\s*|[-•*]\s*)", "", line).strip()
        line = re.sub(r"^answer\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        if set(line.lower().split()) & _NOISE_TOKENS:
            continue
        if 4 < len(line) < 80:
            results.append(line)
        if len(results) >= max_items:
            break
    return results


# ─────────────────────────────────────────────────────────────
# Rule-based last-resort fallback
# ─────────────────────────────────────────────────────────────
_RULE_PATTERNS: Dict[str, List[str]] = {
    "wbc count_LOW":                        ["Leukopenia", "Viral infection"],
    "platelet count_LOW":                   ["Thrombocytopenia", "Dengue fever"],
    "dengue antigen (ns1)- serum_POSITIVE": ["Dengue fever (NS1+)", "Acute dengue infection"],
    "eosinophils_LOW":                      ["Eosinopenia", "Acute stress response"],
    "rbc count_HIGH":                       ["Erythrocytosis", "Dehydration"],
    "packed cell volume [pcv]_HIGH":        ["Relative polycythemia", "Dehydration"],
}


def _rule_fallback(abnormal_values: Dict) -> List[str]:
    seen: set = set()
    out:  List[str] = []
    for raw_name, data in abnormal_values.items():
        status = str(data.get("status", "")).upper()
        key    = f"{raw_name.lower().strip()}_{status}"
        for p in _RULE_PATTERNS.get(key, []):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out[:3] if out else ["Abnormal lab findings — clinical correlation required"]


# ─────────────────────────────────────────────────────────────
# LLM callers
# ─────────────────────────────────────────────────────────────

def _call_biollm(signal_text: str) -> Optional[List[str]]:
    try:
        client   = get_biollm_client()
        prompt   = _PROMPT.format(signals=signal_text)
        log.info("Calling BioLLM for pattern interpretation...")
        resp     = client.text_generation(prompt, max_new_tokens=120, temperature=0.2)
        patterns = _parse_patterns(resp)
        if patterns:
            return patterns
        log.warning("BioLLM output unparseable: %s", resp[:200])
        return None
    except Exception as e:
        log.warning("BioLLM call failed: %s", e)
        return None


def _call_gemini(signal_text: str) -> Optional[List[str]]:
    try:
        prompt   = _PROMPT.format(signals=signal_text)
        log.info("Calling Gemini (fallback) for pattern interpretation...")
        patterns = _parse_patterns(call_gemini(prompt))
        return patterns if patterns else None
    except Exception as e:
        log.error("Gemini pattern call failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────
# Node entry point
# ─────────────────────────────────────────────────────────────

def biollm_direct_node(state: Dict) -> Dict:
    log.info("=== biollm_direct_node: START ===")
    abnormal_values: Dict = state.get("abnormal_values", {})

    if not abnormal_values:
        return {**state, "patterns": ["No abnormal findings detected."], "pattern_engine": "none"}

    signal_text = _compress_signals(abnormal_values)
    log.debug("Compressed signals:\n%s", signal_text)

    patterns = _call_biollm(signal_text)
    engine   = "biollm"

    if not patterns:
        log.warning("BioLLM failed — falling back to Gemini.")
        print("\n⚠️  [PATTERN NODE] BioLLM unavailable. Falling back to Gemini 2.5 Flash.\n")
        patterns = _call_gemini(signal_text)
        engine   = "gemini"

    if not patterns:
        log.error("Gemini also failed — using rule-based fallback.")
        print("\n⚠️  [PATTERN NODE] Both AI backends failed. Using rule-based fallback.\n")
        patterns = _rule_fallback(abnormal_values)
        engine   = "rules_only"

    log.info("=== biollm_direct_node: DONE — engine=%s, patterns=%s ===", engine, patterns)
    return {**state, "patterns": patterns, "pattern_engine": engine}