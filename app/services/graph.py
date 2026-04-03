"""
Medical Report Analysis Graph
==============================
Pipeline: ingest → abnormal → explanation → pattern → safety → summary

File: app/graph.py
"""

import logging
import traceback
from typing import TypedDict, Dict, List

from langgraph.graph import StateGraph

from app.services.ingest_node        import ingest_node
from app.services.abnormal_node      import abnormal_node
from app.services.explanation_node   import explanation_node
from app.services.biollm_direct_node import biollm_direct_node
from app.services.safety_node        import safety_node
from app.services.summary_node       import summary_node

# ─────────────────────────────────────────────
# 📋 LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DIVIDER  = "=" * 65
THIN_DIV = "─" * 65
_RISK    = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}
_ENG     = {
    "biollm":     "BioLLM (Llama3-OpenBioLLM-8B)",
    "gemini":     "Gemini 2.5 Flash  ⚠️  (BioLLM fallback)",
    "rules_only": "Rule-based  ⚠️  (both AI backends failed)",
    "none":       "None",
}

def _log_section(title: str):
    log.info(DIVIDER)
    log.info(f"  🔷  {title}")
    log.info(DIVIDER)

def _log_step(msg: str):
    log.info(f"  ➤  {msg}")

def _log_ok(msg: str):
    log.info(f"  ✅  {msg}")

def _log_warn(msg: str):
    log.warning(f"  ⚠️   {msg}")

def _log_err(msg: str, exc: Exception = None):
    log.error(f"  ❌  {msg}")
    if exc:
        log.error(f"  📛  {type(exc).__name__}: {exc}")
        log.error("  📛  Traceback:\n" + traceback.format_exc())


# ─────────────────────────────────────────────
# 🧩 STATE SCHEMA
# ─────────────────────────────────────────────
class MedicalState(TypedDict):
    file_path:         str
    raw_text:          str
    structured_report: Dict
    warnings:          List[str]
    abnormal_values:   Dict
    explanation:       Dict
    patterns:          List[str]
    pattern_engine:    str
    final_output:      Dict


# ─────────────────────────────────────────────
# 🔭 NODE WRAPPERS  – every node is wrapped so
#    we get identical enter/exit logs + error
#    trapping for every step automatically.
# ─────────────────────────────────────────────
def _wrap(name: str, fn):
    """Return a version of `fn` that logs entry, output, and any exception."""
    def wrapper(state: MedicalState) -> MedicalState:
        _log_section(f"NODE: {name.upper()}")

        # ── Show what's coming in ──────────────────────────────
        _log_step("State entering this node:")
        for k, v in state.items():
            if v is None or v == "" or v == {} or v == []:
                _log_step(f"  {k:20s}: (empty)")
            elif isinstance(v, str):
                snippet = v[:120].replace("\n", " ")
                _log_step(f"  {k:20s}: {snippet}{'…' if len(v) > 120 else ''}")
            elif isinstance(v, dict):
                _log_step(f"  {k:20s}: dict with keys {list(v.keys())}")
            elif isinstance(v, list):
                _log_step(f"  {k:20s}: list [{len(v)} items] {v[:3]}{'…' if len(v) > 3 else ''}")
            else:
                _log_step(f"  {k:20s}: {v}")

        # ── Run the real node ──────────────────────────────────
        _log_step(f"Calling {fn.__name__}() ...")
        try:
            output = fn(state)
        except Exception as exc:
            _log_err(f"Node '{name}' RAISED AN EXCEPTION — pipeline will likely halt.", exc)
            raise  # re-raise so LangGraph can handle it

        # ── Show what came out ─────────────────────────────────
        _log_ok(f"Node '{name}' finished without exception.")
        _log_step("Output produced by this node:")

        if not isinstance(output, dict):
            _log_warn(f"Unexpected output type: {type(output)} — value: {output}")
        else:
            for k, v in output.items():
                if v is None or v == "" or v == {} or v == []:
                    _log_step(f"  {k:20s}: (empty)")
                elif isinstance(v, str):
                    snippet = v[:200].replace("\n", " ")
                    _log_step(f"  {k:20s}: {snippet}{'…' if len(v) > 200 else ''}")
                elif isinstance(v, dict):
                    _log_step(f"  {k:20s}: dict {list(v.keys())}")
                    for dk, dv in v.items():
                        _log_step(f"      └─ {dk}: {str(dv)[:100]}")
                elif isinstance(v, list):
                    _log_step(f"  {k:20s}: list [{len(v)} items]")
                    for i, item in enumerate(v[:5]):
                        _log_step(f"      [{i}] {str(item)[:120]}")
                    if len(v) > 5:
                        _log_step(f"      … {len(v) - 5} more items")
                else:
                    _log_step(f"  {k:20s}: {v}")

        log.info(THIN_DIV)
        return output

    wrapper.__name__ = fn.__name__
    return wrapper


# ─────────────────────────────────────────────
# 🏗  BUILD GRAPH
# ─────────────────────────────────────────────
def build_graph():
    _log_section("BUILDING LANGGRAPH PIPELINE")

    builder = StateGraph(MedicalState)

    nodes = [
        ("ingest",      ingest_node),
        ("abnormal",    abnormal_node),
        ("explanation", explanation_node),
        ("pattern",     biollm_direct_node),
        ("safety",      safety_node),
        ("summary",     summary_node),
    ]

    for node_name, node_fn in nodes:
        wrapped = _wrap(node_name, node_fn)
        builder.add_node(node_name, wrapped)
        _log_step(f"Registered node: '{node_name}' → {node_fn.__name__}()")

    builder.set_entry_point("ingest")
    builder.add_edge("ingest",      "abnormal")
    builder.add_edge("abnormal",    "explanation")
    builder.add_edge("explanation", "pattern")
    builder.add_edge("pattern",     "safety")
    builder.add_edge("safety",      "summary")

    _log_step("Edges: ingest→abnormal→explanation→pattern→safety→summary")

    compiled = builder.compile()
    _log_ok("Pipeline compiled successfully.")
    log.info(THIN_DIV)
    return compiled


graph = build_graph()


# ─────────────────────────────────────────────
# 🖨  PRETTY PRINTER
# ─────────────────────────────────────────────
def _print_report(result: Dict) -> None:
    report      = result.get("structured_report", {})
    patient     = report.get("patient_info", {})
    abnormal    = result.get("abnormal_values",  {})
    explanation = result.get("explanation",      {})
    patterns    = result.get("patterns",         [])
    p_engine    = result.get("pattern_engine",   "unknown")
    final       = result.get("final_output",     {})
    s_engine    = final.get("summary_engine",    "unknown")

    print(f"\n{'='*65}\n  🏥  MEDICAL REPORT ANALYSIS\n{'='*65}")

    print(f"\n👤  PATIENT\n{THIN_DIV}")
    print(f"  Name    : {patient.get('name','N/A')}")
    print(f"  Age/Sex : {patient.get('age','N/A')} / {patient.get('gender','N/A')}")
    print(f"  Date    : {patient.get('report_date','N/A')}")

    print(f"\n📊  TEST RESULTS\n{THIN_DIV}")
    print(f"  {'TEST':<40} {'VALUE':<12} STATUS")
    print(THIN_DIV)
    for t in report.get("test_results", []):
        status = t.get("status", "")
        icon   = "⚠️ " if status in ("LOW", "HIGH", "POSITIVE", "REACTIVE") else "✅ "
        print(f"  {icon}{t['test']:<38} {str(t['value']):<12} {status}")

    if abnormal:
        print(f"\n🚨  ABNORMAL VALUES + CLINICAL EXPLANATION\n{THIN_DIV}")
        for test, data in abnormal.items():
            exp = explanation.get(test, "—")
            print(f"\n  {test}  [{data.get('status','')}]  "
                  f"value={data.get('value','')}  ref={data.get('range','N/A')}")
            print(f"  💬 {exp}")

    print(f"\n🔬  DETECTED PATTERNS  [{_ENG.get(p_engine, p_engine)}]\n{THIN_DIV}")
    for p in patterns:
        print(f"  • {p}")

    risk_icon = _RISK.get(final.get("risk", ""), "⚪")
    print(f"\n🧾  CLINICAL SUMMARY  [{_ENG.get(s_engine, s_engine)}]\n{THIN_DIV}")
    print(f"\n  📝  Summary:\n  {final.get('summary','N/A')}")
    print(f"\n  🔍  Interpretation:\n  {final.get('interpretation','N/A')}")
    print(f"\n  {risk_icon}  Risk Level: {final.get('risk','N/A')}")

    alerts = final.get("alerts", [])
    if alerts:
        print(f"\n  ⚠️   Alerts:")
        for a in alerts:
            print(f"      - {a}")

    print(f"\n  💊  Recommendation:\n  {final.get('recommendation','N/A')}")
    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────
# 🧪  CLI TEST RUNNER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    TEST_PDF = r"..\..\downloads\Investigation.pdf"

    # ── Full invoke ────────────────────────────────────────────
    _log_section("CLI TEST — graph.invoke()")
    _log_step(f"Input file: {TEST_PDF}")
    try:
        result = graph.invoke({"file_path": TEST_PDF})
        _log_ok("graph.invoke() completed.")
        _print_report(result)
    except Exception as e:
        _log_err("graph.invoke() failed.", e)
        sys.exit(1)

