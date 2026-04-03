"""
Explanation Node
=================
Generates a clean one-sentence clinical explanation per abnormal value.

Priority chain per test:
  1. Rule-based lookup  (instant, zero API cost)
  2. BioLLM
  3. Gemini fallback
  4. Generic safe string

File: app/services/explanation_node.py
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from app.services.llm_clients import call_gemini, get_biollm_client

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Rule-based lookup (covers the most common tests — no API call)
# ─────────────────────────────────────────────────────────────
_RULES: Dict[tuple, str] = {
    ("wbc count",                "LOW"):  "Leukopenia indicates a reduced white blood cell count, raising concern for immune suppression or bone marrow dysfunction.",
    ("wbc count",                "HIGH"): "Leukocytosis reflects an elevated white blood cell count, commonly associated with infection, inflammation, or a stress response.",
    ("hemoglobin",               "LOW"):  "Anemia is indicated by a below-normal hemoglobin level, reducing the blood's oxygen-carrying capacity.",
    ("hemoglobin",               "HIGH"): "Polycythemia is suggested by an elevated hemoglobin level, potentially due to chronic hypoxia or primary polycythemia vera.",
    ("rbc count",                "LOW"):  "Erythropenia reflects a reduced red blood cell count, commonly associated with anemia or bone marrow suppression.",
    ("rbc count",                "HIGH"): "Erythrocytosis indicates an elevated RBC count, possibly secondary to dehydration, high altitude adaptation, or polycythemia.",
    ("platelet count",           "LOW"):  "Thrombocytopenia reflects a reduced platelet count, significantly increasing the risk of bleeding and bruising.",
    ("platelet count",           "HIGH"): "Thrombocytosis indicates an elevated platelet count, which may be reactive (infection/inflammation) or primary (myeloproliferative disorder).",
    ("packed cell volume [pcv]", "HIGH"): "Elevated PCV suggests increased red cell mass, consistent with polycythemia, dehydration, or secondary erythrocytosis.",
    ("packed cell volume [pcv]", "LOW"):  "Reduced PCV indicates a decreased proportion of red blood cells, consistent with anemia.",
    ("eosinophils",              "LOW"):  "Eosinopenia may reflect an acute stress response, corticosteroid effect, or early systemic infection suppressing eosinophil production.",
    ("eosinophils",              "HIGH"): "Eosinophilia is associated with allergic conditions, parasitic infections, or autoimmune disorders.",
    ("esr",                      "HIGH"): "An elevated ESR is a non-specific marker of systemic inflammation, infection, or autoimmune activity.",
    ("crp",                      "HIGH"): "Elevated CRP indicates an acute-phase inflammatory response, commonly triggered by infection, tissue injury, or systemic inflammation.",
    ("dengue antigen (ns1)- serum", "POSITIVE"): "A positive Dengue NS1 antigen confirms active dengue virus infection, typically detectable in the first few days of illness.",
    ("dengue-igm antibodies- serum", "POSITIVE"): "Positive Dengue IgM antibodies indicate an acute or recent dengue infection, typically appearing 3–5 days after symptom onset.",
    ("dengue-igg antibodies- serum", "POSITIVE"): "Positive Dengue IgG antibodies suggest past dengue exposure or a secondary dengue infection with potential risk of severe disease.",
}

# Noise prefix patterns to strip from any LLM output
_NOISE = re.compile(
    r"^(may suggest[,:]?\s*|the answer (may be|is)[,:]?\s*|this may\s*|"
    r"note[,:]?\s*|output[,:]?\s*|answer[,:]?\s*|explanation[,:]?\s*)",
    re.IGNORECASE,
)

_PROMPT = """\
You are a clinical medical expert.

Lab result: {test_name} is {status} (value: {value}, reference range: {range}).

Write exactly ONE sentence explaining the clinical significance of this result.

STRICT RULES:
- Begin directly with the medical fact. No preamble.
- Do NOT start with "May suggest", "This may", "The result", "Note:", or any introduction.
- Do NOT repeat the test name or the numeric value.
- One sentence only. Proper medical terminology.

Correct examples:
  Leukopenia indicates reduced white blood cells, raising concern for immune suppression or bone marrow pathology.
  Thrombocytopenia reflects a reduced platelet count, increasing the risk of spontaneous bleeding.

Answer:"""



def _clean(text: str) -> str:
    text = text.strip()
    text = _NOISE.sub("", text).strip()
    # Take only the first sentence if model output leaked more
    m = re.match(r"^(.+?[.!?])(\s|$)", text)
    if m:
        text = m.group(1)
    return text[0].upper() + text[1:] if text else text


def _biollm_explain(test_name: str, status: str, value: str, range_str: str) -> Optional[str]:
    try:
        client = get_biollm_client()
        prompt = _PROMPT.format(test_name=test_name, status=status, value=value, range=range_str)
        resp   = client.text_generation(prompt, max_new_tokens=80, temperature=0.2)
        cleaned = _clean(resp)
        return cleaned if len(cleaned) > 20 else None
    except Exception as e:
        log.warning("BioLLM explanation failed for '%s': %s", test_name, e)
        return None


def _gemini_explain(test_name: str, status: str, value: str, range_str: str) -> Optional[str]:
    try:
        prompt  = _PROMPT.format(test_name=test_name, status=status, value=value, range=range_str)
        cleaned = _clean(call_gemini(prompt))
        return cleaned if len(cleaned) > 20 else None
    except Exception as e:
        log.error("Gemini explanation failed for '%s': %s", test_name, e)
        return None


def _explain_one(test_name: str, data: Dict) -> str:
    status    = str(data.get("status", "")).upper()
    value     = str(data.get("value",  ""))
    range_str = str(data.get("range",  "N/A"))

    # 1 — rule (fastest, cleanest)
    rule = _RULES.get((test_name.lower().strip(), status))
    if rule:
        return rule

    # 2 — BioLLM
    log.info("Calling BioLLM for explanation: %s", test_name)
    result = _biollm_explain(test_name, status, value, range_str)
    if result:
        return result

    # 3 — Gemini fallback
    log.warning("BioLLM failed for '%s' — falling back to Gemini.", test_name)
    print(f"\n⚠️  [EXPLANATION NODE] BioLLM failed for '{test_name}'. Using Gemini.\n")
    result = _gemini_explain(test_name, status, value, range_str)
    if result:
        return result

    # 4 — Generic safe fallback
    return f"{test_name} is {status.lower()} — clinical correlation is recommended."


def explanation_node(state: Dict) -> Dict:
    log.info("=== explanation_node: START ===")
    abnormal_values: Dict = state.get("abnormal_values", {})
    explanation: Dict = {}
    for test_name, data in abnormal_values.items():
        explanation[test_name] = _explain_one(test_name, data)
        log.info("Explained: %s", test_name)
    log.info("=== explanation_node: DONE — %d explanations ===", len(explanation))
    return {**state, "explanation": explanation}