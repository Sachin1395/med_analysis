import os
import requests
import uuid
import mimetypes
import logging
import traceback
import time

from fastapi.responses import StreamingResponse
import asyncio
import collections


from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

from app.services.graph import build_graph
from app.services.llm_clients import call_gemini

from fastapi.responses import HTMLResponse, JSONResponse
import math, time


# ─────────────────────────────────────────────
# 📋 LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DIVIDER     = "=" * 65
THIN_DIV    = "─" * 65
STEP_PREFIX = "  ➤"

log_buffer: collections.deque = collections.deque(maxlen=200)

class LandingPageLogHandler(logging.Handler):
    """Captures log records into the ring buffer for the landing page."""
    def emit(self, record):
        msg = self.format(record)
        log_buffer.append({
            "ts":  record.asctime if hasattr(record, 'asctime') else "",
            "level": record.levelname,
            "msg": record.getMessage(),
        })

# Attach the handler to your existing logger
_lp_handler = LandingPageLogHandler()
_lp_handler.setFormatter(logging.Formatter("%(asctime)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_lp_handler)


def log_section(title: str):
    log.info(DIVIDER)
    log.info(f"  🔷  {title}")
    log.info(DIVIDER)

def log_step(msg: str):
    log.info(f"{STEP_PREFIX} {msg}")

def log_success(msg: str):
    log.info(f"  ✅  {msg}")

def log_warning(msg: str):
    log.warning(f"  ⚠️   {msg}")

def log_failure(msg: str, exc: Exception = None):
    log.error(f"  ❌  {msg}")
    if exc:
        log.error(f"  📛  Exception type : {type(exc).__name__}")
        log.error(f"  📛  Exception detail: {exc}")
        log.error("  📛  Full traceback:\n" + traceback.format_exc())


# ─────────────────────────────────────────────
# 🚀 APP + CLIENTS
# ─────────────────────────────────────────────
app = FastAPI()

TWILIO_SID        = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

UPLOAD_DIR = "downloads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_TTL_SECONDS = 24 * 60 * 60  # 24 hours


# ─────────────────────────────────────────────
# 🧠 SESSION MEMORY  (Step 1 & 2)
# ─────────────────────────────────────────────
# Structure per user:
# {
#   "phone_number": {
#       "report_context": {...},   # clean compressed context
#       "timestamp": float,        # unix time of upload
#       "pending_replace": bool,   # waiting for user confirm?
#       "pending_file_path": str,  # queued new file (if confirming)
#   }
# }
user_sessions: dict[str, dict] = {}


def _is_session_alive(phone: str) -> bool:
    """Return True if the user has an active, non-expired report session."""
    session = user_sessions.get(phone)
    if not session:
        return False
    age = time.time() - session.get("timestamp", 0)
    if age > SESSION_TTL_SECONDS:
        log_warning(f"Session expired for {phone} (age={age:.0f}s) — clearing.")
        del user_sessions[phone]
        return False
    return True


def _build_report_context(result: dict) -> dict:
    """
    Step 5 & 6: Extract ONLY the useful fields from LangGraph output.
    Excludes raw OCR text and noisy metadata.
    """
    structured = result.get("structured_report", {})
    abnormal   = result.get("abnormal_values",   {})
    explanation= result.get("explanation",        {})
    patterns   = result.get("patterns",           [])
    final      = result.get("final_output",       {})

    context = {
        "structured_report": structured,
        "abnormal_values":   abnormal,
        "explanation":       explanation,
        "patterns":          patterns,
        "summary":           final.get("summary", "") if isinstance(final, dict) else str(final),
        "risk_level":        final.get("risk",    "Unknown") if isinstance(final, dict) else "Unknown",
        "alerts":            final.get("alerts",  []) if isinstance(final, dict) else [],
    }
    log_step(f"Built report context — keys: {list(context.keys())}")
    return context


def _save_session(phone: str, report_context: dict):
    """Persist clean report context for a user."""
    user_sessions[phone] = {
        "report_context":  report_context,
        "timestamp":       time.time(),
        "pending_replace": False,
        "pending_file_path": None,
    }
    log_success(f"Session saved for {phone}.")


def _clear_pending(phone: str):
    """Cancel a pending replacement confirmation."""
    if phone in user_sessions:
        user_sessions[phone]["pending_replace"]   = False
        user_sessions[phone]["pending_file_path"] = None


# ─────────────────────────────────────────────
# 🤖 GEMINI HELPERS
# ─────────────────────────────────────────────

def chat_with_gemini(user_input: str) -> str:
    """
    Before-report mode gatekeeper — only allows upload-related conversation.
    NOTE: This function must NEVER be called when a session already exists.
    Branch C in the webhook handles all session-aware traffic.
    """
    log_section("GEMINI CHAT  [no-report mode]")
    log_step(f"User input: {user_input!r}")

    prompt = f"""
You are a medical report assistant chatbot.
The user has NOT uploaded any report yet.
STRICT RULES:
- ONLY help with uploading and analyzing medical reports
- DO NOT answer general medical questions or food/diet questions
- DO NOT diagnose
- Politely tell the user to upload their medical report first
- Be friendly and short

User message:
{user_input}

Response:
"""
    try:
        response = call_gemini(prompt)
        log_step(f"Raw Gemini response:\n{THIN_DIV}\n{response.strip()}\n{THIN_DIV}")

        lowered = response.lower()
        # Block hard medical advice slipping through
        if any(x in lowered for x in ["you have", "you are diagnosed", "prescribe", "take this medicine"]):
            log_warning("Response contained restricted medical terms — overriding.")
            return "⚠️ I can only help analyze medical reports. Please upload your report."

        log_success("Gemini gatekeeper response accepted.")
        return response.strip()

    except Exception as e:
        log_failure("Gemini chat call failed.", e)
        return "⚠️ Unable to respond. Please upload your medical report."


def _classify_question(user_input: str) -> str:
    """
    Lightweight keyword classifier — routes the question to the right
    prompt template BEFORE calling Gemini, so we never waste a call.

    Categories:
      MEDICINE   → asking for drugs / medications
      DIAGNOSE   → asking for a definitive diagnosis
      VALUE      → asking what a specific lab value means
      CONDITION  → asking what a medical term / condition means
      RISK       → asking about severity / seriousness
      NEXTSTEPS  → asking what to do next / which doctor
      LIFESTYLE  → asking about precautions / rest / diet
      CLARIFY    → asking which value is worst / causes / connections
      GENERAL    → anything else report-related
    """
    txt = user_input.lower()

    medicine_kw   = ["medicine", "medication", "drug", "tablet", "tablet", "pill",
                     "paracetamol", "antibiotic", "dose", "prescription", "treat at home",
                     "home remedy", "what to take", "should i take"]
    diagnose_kw   = ["do i have", "is it confirmed", "for sure", "definitely have",
                     "am i positive", "is this dengue", "diagnos"]
    nextsteps_kw  = ["what should i do", "next step", "see a doctor", "consult",
                     "which doctor", "specialist", "need more test", "go to hospital",
                     "emergency", "immediately"]
    lifestyle_kw  = ["precaution", "rest", "avoid", "normal activit", "diet",
                     "food", "drink", "fluid", "water", "exercise", "sleep", "continue",
                     # eating / drinking patterns
                     "can i eat", "can i drink", "can i have", "is it ok to eat",
                     "is it safe to eat", "should i eat", "what to eat", "what can i eat",
                     "what should i eat", "what can i drink", "what to drink",
                     # specific food/drink items users commonly ask about
                     "milk", "juice", "coffee", "tea", "rice", "fruit", "vegetable",
                     "sugar", "sweet", "spicy", "junk", "fast food", "snack",
                     "meal", "breakfast", "lunch", "dinner",
                     # activity / lifestyle
                     "walk", "gym", "sport", "run", "bath", "shower", "work", "office",
                     "travel", "go out", "outdoor", "screen", "phone"]
    risk_kw       = ["serious", "severity", "dangerous", "severe", "risk",
                     "worry", "concern", "complication", "critical", "moderate", "high risk"]
    condition_kw  = ["what is", "what does", "thrombocytopenia", "polycythemia",
                     "eosinophil", "leukopenia", "immune", "ns1", "antigen", "dengue fever",
                     "packed cell", "mean corpuscular", "hematocrit", "hemoglobin",
                     "platelet", "white blood", "red blood", "creatinine", "bilirubin"]
    clarify_kw    = ["most concern", "which value", "causing", "temporary", "connected",
                     "related", "infection", "all finding", "overall", "explain"]

    if any(k in txt for k in medicine_kw):
        return "MEDICINE"
    if any(k in txt for k in diagnose_kw):
        return "DIAGNOSE"
    if any(k in txt for k in nextsteps_kw):
        return "NEXTSTEPS"
    if any(k in txt for k in lifestyle_kw):
        return "LIFESTYLE"
    if any(k in txt for k in risk_kw):
        return "RISK"
    if any(k in txt for k in condition_kw):
        return "CONDITION"
    if any(k in txt for k in clarify_kw):
        return "CLARIFY"
    return "GENERAL"


# Pre-built safe responses for hard-blocked categories
_MEDICINE_BLOCK = (
    "💊 I'm not able to suggest any medicines or treatments.\n\n"
    "Every person's body is different, and the wrong medicine can be harmful.\n\n"
    "👉 *Please consult your doctor* — they will prescribe exactly what is right for you."
)

_DIAGNOSE_BLOCK = (
    "🔬 I cannot confirm a diagnosis — only a qualified doctor can do that.\n\n"
    "What I *can* tell you is what your report results *suggest*.\n"
    "Based on your report:\n"
    "{summary}\n\n"
    "👉 Please visit a doctor who can examine you and confirm the actual condition."
)


def chat_with_report_context(user_input: str, report_context: dict) -> str:
    """
    Full category-aware follow-up chat.
    - Classifies intent before calling Gemini
    - Hard-blocks medicine & diagnosis requests
    - Sends category-specific prompts for all other questions
    - Enforces plain language (understandable by anyone, even a child)
    - Safe lifestyle/next-steps advice is always allowed
    """
    log_section("GEMINI CHAT  [follow-up / report-aware mode]")
    log_step(f"User input: {user_input!r}")

    category = _classify_question(user_input)
    log_step(f"Classified as: {category}")

    # ── Hard blocks (no Gemini call needed) ──────────────────────
    if category == "MEDICINE":
        log_warning("Medicine request detected — returning hard block.")
        return _MEDICINE_BLOCK

    if category == "DIAGNOSE":
        log_warning("Diagnosis confirmation request — returning safe block.")
        summary = report_context.get("summary", "Your report has some abnormal values.")
        return _DIAGNOSE_BLOCK.format(summary=summary)

    # ── Build shared report data snippet ─────────────────────────
    report_snippet = f"""
REPORT DATA (use ONLY this to answer):
- Summary         : {report_context.get('summary',        'N/A')}
- Risk Level      : {report_context.get('risk_level',     'N/A')}
- Abnormal Values : {report_context.get('abnormal_values',{})}
- Explanations    : {report_context.get('explanation',    {})}
- Detected Patterns: {report_context.get('patterns',      [])}
- Alerts          : {report_context.get('alerts',         [])}
"""

    # ── Category-specific prompt templates ───────────────────────
    if category == "VALUE":
        task = """
The patient is asking what a specific lab value in their report means.

INSTRUCTIONS:
- Explain what that value normally does in the body IN ONE SIMPLE SENTENCE (like explaining to a 10-year-old).
- Then state what their result shows (high / low / normal).
- Then say what that *might suggest* based on the report — use "may suggest" or "could indicate".
- DO NOT diagnose. DO NOT prescribe.
- End with: "Your doctor can explain this better after examining you."
"""

    elif category == "CONDITION":
        task = """
The patient is asking what a medical term or condition name means.

INSTRUCTIONS:
- Explain the term in the SIMPLEST possible language — as if talking to a child.
  Example: "Thrombocytopenia just means your blood has fewer platelets than normal.
            Platelets are tiny cells that help your blood clot when you get a cut."
- Then connect it to what their report actually shows.
- DO NOT diagnose. DO NOT prescribe.
- End with: "Your doctor will explain what this means for you specifically."
"""

    elif category == "RISK":
        task = """
The patient is asking how serious their condition is.

INSTRUCTIONS:
- State the risk level from the report clearly (High / Moderate / Low).
- Explain in simple words WHY it is at that level based on the abnormal values.
- List any alerts from the report as bullet points.
- Use calm, reassuring language — do NOT alarm the patient unnecessarily.
- DO NOT diagnose. DO NOT say they definitely have any disease.
- End with: "Please consult a doctor as soon as possible to get a proper evaluation."
"""

    elif category == "NEXTSTEPS":
        task = """
The patient is asking what they should do next.

INSTRUCTIONS:
- Give clear, practical next steps based on the risk level and alerts in the report.
- If risk is High → say "Please see a doctor TODAY or go to a hospital."
- If risk is Moderate → say "See a doctor within the next 1–2 days."
- If risk is Low → say "Monitor your symptoms and follow up with a doctor soon."
- Mention which type of specialist may be relevant IF it is obvious from the report
  (e.g., hematologist for blood disorders) — but say "your doctor will refer you."
- DO NOT prescribe. DO NOT diagnose.
"""

    elif category == "LIFESTYLE":
        task = """
The patient is asking about lifestyle, precautions, diet, rest, or daily activities.

INSTRUCTIONS:
- Give safe, natural, harmless general advice based on the report findings.
- For LOW PLATELET COUNT specifically:
    • "Drink plenty of fluids — water, coconut water, or fresh juices."
    • "Rest as much as possible. Avoid heavy physical activity."
    • "Avoid blood-thinning foods like alcohol."
    • "Watch for new symptoms like unusual bruising or bleeding."
    • "See a doctor soon — do not wait."
- For other findings, give similarly simple, safe general guidance.
- NEVER suggest specific medicines, supplements, or herbal remedies.
- NEVER claim these steps will cure or treat any condition.
- Always end with: "These are general precautions only. Your doctor's advice comes first."
"""

    elif category == "CLARIFY":
        task = """
The patient wants a clearer overall picture — e.g., which value is most concerning,
are findings connected, or what is causing the abnormalities.

INSTRUCTIONS:
- Pick the most abnormal value from the report and explain why it stands out.
- Briefly explain how the findings may be connected in simple words.
  Example: "When you have an infection, your body fights it — this can cause
            platelet counts to drop and white blood cells to change."
- Use cause-and-effect language a child could follow.
- DO NOT diagnose. Use "may be", "could suggest", "might mean".
- End with: "A doctor can confirm the exact cause after a physical exam."
"""

    else:  # GENERAL
        task = """
The patient has a general question about their report or about what they can do / eat / drink.

INSTRUCTIONS:
- If the question is about food, drinks, or daily activities:
    • Answer with safe, general, friendly guidance based on their report findings.
    • Use simple examples (e.g., "Rose milk has sugar — since your glucose is high, try to have it rarely or in small amounts").
    • NEVER suggest specific medicines or supplements.
    • Always end with: "These are general tips only. Your doctor's advice comes first."
- If the question is about the report itself:
    • Answer based ONLY on the report data provided.
    • If it cannot be answered from the report, say:
      "I can only answer questions based on your uploaded report. This goes beyond what your report shows."
- Use the simplest possible language — no medical jargon, short sentences.
- DO NOT diagnose. DO NOT prescribe.
- Always end with a reminder to consult a doctor.
"""

    prompt = f"""
You are a friendly medical report assistant explaining a lab report to a patient on WhatsApp.

{report_snippet}

YOUR TASK:
{task}

LANGUAGE RULES (VERY IMPORTANT):
- Write like you are talking to a 10-year-old. No medical jargon.
- Short sentences. Simple words.
- Use analogies if helpful. Example: "Platelets are like tiny band-aids in your blood."
- Use bullet points (•) for lists — never numbered lists.
- Keep the total response under 200 words.
- Use WhatsApp formatting: *bold* for key terms, _italic_ for mild emphasis.

Patient's question:
{user_input}

Response:
"""

    try:
        response = call_gemini(prompt)
        log_step(f"Raw follow-up response (400 chars):\n{response[:400]}")

        # Post-generation safety sweep
        forbidden_phrases = [
            "you have dengue", "you are infected", "you are diagnosed",
            "you suffer from", "take this medicine", "prescribe",
            "you definitely have", "confirmed diagnosis"
        ]
        lowered = response.lower()
        if any(fp in lowered for fp in forbidden_phrases):
            log_warning("Forbidden diagnostic language detected in response — overriding.")
            return (
                "⚠️ I can explain what your report *suggests*, but I cannot confirm "
                "any diagnosis.\n\n"
                "👉 Please consult a doctor — they are the only ones who can tell "
                "you what you actually have."
            )

        cleaned = clean_whatsapp_output(response)
        log_success(f"Follow-up response accepted. Category={category}, Length={len(cleaned)}")
        return cleaned

    except Exception as e:
        log_failure("Gemini follow-up chat call failed.", e)
        return "⚠️ Unable to answer right now. Please try again in a moment."


def clean_whatsapp_output(text: str) -> str:
    lines_in = text.splitlines()
    cleaned  = [line.strip() for line in lines_in if line.strip() and "```" not in line]
    log_step(f"clean_whatsapp_output: {len(lines_in)} raw lines → {len(cleaned)} clean lines")
    return "\n".join(cleaned)


def format_output_for_user(data: dict) -> str | None:
    log_section("FORMATTING REPORT FOR WHATSAPP")
    log_step(f"Input data keys: {list(data.keys())}")

    prompt = f"""
You are a medical assistant explaining lab reports.
Convert the following data into WhatsApp-friendly format.

DATA:
{data}

FORMAT STRICTLY LIKE THIS:
*🧾 Medical Report Summary*

*📌 Summary:*
...

*What it means:*
...

*Risk Level:* High / Moderate / Low

*Important Alerts:*
• ...

*What you should do:*
...
"""
    try:
        response  = call_gemini(prompt)
        log_step(f"Raw formatting response (first 400 chars):\n{response[:400]}")
        formatted = clean_whatsapp_output(response)
        log_success("Report formatted successfully.")
        return formatted
    except Exception as e:
        log_failure("Gemini formatting call failed.", e)
        return None


def download_media(url: str, content_type: str) -> str | None:
    log_section("DOWNLOADING MEDIA")
    log_step(f"URL          : {url}")
    log_step(f"Content-Type : {content_type}")

    ext       = mimetypes.guess_extension(content_type) or ".bin"
    filename  = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    log_step(f"Target path  : {file_path}")

    try:
        response = requests.get(url, auth=(TWILIO_SID, TWILIO_AUTH_TOKEN), timeout=15)
        log_step(f"HTTP status  : {response.status_code}")

        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            log_success(f"File saved → {file_path}")
            return filename

        log_failure(f"Download rejected — HTTP {response.status_code}. Body: {response.text[:200]}")

    except requests.exceptions.Timeout:
        log_failure("Download timed out after 15 s.")
    except Exception as e:
        log_failure("Unexpected download error.", e)

    return None


# ─────────────────────────────────────────────
# ⏳ BACKGROUND WORKER
# ─────────────────────────────────────────────
def process_report_task(file_path: str, from_number: str, to_number: str):
    log_section(f"BACKGROUND REPORT PROCESSOR  |  from={from_number}")
    log_step(f"File path   : {file_path}")

    # ── Step 1: Run LangGraph ───────────────────────────────────
    log_step("STEP 1 — Invoking LangGraph pipeline...")
    try:
        result = graph.invoke({"file_path": file_path})
        log_success("LangGraph pipeline completed.")
    except Exception as e:
        log_failure("LangGraph pipeline crashed.", e)
        _send_whatsapp(
            "❌ Sorry, there was an error processing your report. Please try again.",
            from_number, to_number
        )
        return

    # ── Step 2: Build and save clean report context ─────────────
    log_step("STEP 2 — Building clean report context...")
    report_context = _build_report_context(result)
    _save_session(from_number, report_context)  # ← persists to in-memory store

    # ── Step 3: Format for WhatsApp ─────────────────────────────
    log_step("STEP 3 — Formatting output for WhatsApp...")
    final_output = result.get("final_output")

    if isinstance(final_output, dict):
        reply = format_output_for_user(final_output)
        if not reply:
            log_warning("Formatting returned None — falling back to raw summary field.")
            reply = f"*Summary:* {final_output.get('summary', 'N/A')}"
    else:
        log_warning("final_output is not a dict — using plain text fallback.")
        reply = f"🧾 Medical Report Analysis\n\n{final_output}"

    # Append structured follow-up question guide
    reply += (
        "\n\n─────────────────────\n"
        "💬 *You can now ask questions about your report!*\n\n"
        "Here are some things you can ask:\n\n"
        "🔬 *About your values:*\n"
        "• Why are my platelets low?\n"
        "• What does my WBC count mean?\n\n"
        "⚠️ *About risk & severity:*\n"
        "• Is my condition serious?\n"
        "• Can this become dangerous?\n\n"
        "🧠 *To understand a term:*\n"
        "• What is thrombocytopenia?\n"
        "• What does NS1 positive mean?\n\n"
        "🩺 *What to do next:*\n"
        "• Should I see a doctor immediately?\n"
        "• What precautions should I take?\n\n"
        "_Just type your question and I'll answer based on your report._"
    )

    log_step(f"Final reply length: {len(reply)} chars")

    # ── Step 4: Send via Twilio ─────────────────────────────────
    log_step("STEP 4 — Sending WhatsApp message via Twilio...")
    _send_whatsapp(reply, from_number, to_number)

    log_section(f"BACKGROUND TASK COMPLETE  |  from={from_number}")

def _send_whatsapp(body: str, from_number: str, to_number: str):
    log_step(f"  To   : {from_number}")
    log_step(f"  From : {to_number}")
    try:
        MAX_LEN = 1500
        chunks = [body[i:i+MAX_LEN] for i in range(0, len(body), MAX_LEN)]

        for chunk in chunks:
            msg = twilio_client.messages.create(
                body=chunk,
                from_=to_number,
                to=from_number
            )
            log_success(f"Message chunk sent. Twilio SID: {msg.sid}")
            if len(chunks) > 1:
                time.sleep(0.5)

    except Exception as e:
        log_failure("Failed to send WhatsApp message via Twilio.", e)
        if hasattr(e, 'code'):
            log_failure(f"Twilio error code: {e.code}")
        if hasattr(e, 'msg'):
            log_failure(f"Twilio error msg: {e.msg}")
        if hasattr(e, 'details'):
            log_failure(f"Twilio error details: {e.details}")
# ─────────────────────────────────────────────
# 📲 WHATSAPP WEBHOOK
# ─────────────────────────────────────────────
log_section("INITIALISING APPLICATION")
log_step("Building LangGraph pipeline...")
try:
    graph = build_graph()
    log_success("LangGraph pipeline built successfully.")
except Exception as e:
    log_failure("Failed to build LangGraph pipeline at startup!", e)
    raise
log.info(THIN_DIV)


# ─────────────────────────────────────────────
# ADD THESE IMPORTS at the top of your main.py
# ─────────────────────────────────────────────
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# import math, time

# ─────────────────────────────────────────────
# 🌐 LANDING PAGE ROUTE  —  GET /
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Serves the MedInsight AI landing page."""
    with open("../landing.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


# ─────────────────────────────────────────────
# 📊 SESSIONS API  —  GET /sessions
# ─────────────────────────────────────────────
@app.get("/sessions")
async def get_sessions():
    """
    Returns all currently active sessions with masked phone numbers.
    Used by the landing page's live sessions panel.

    Response shape:
    {
      "total": 3,
      "sessions": [
        {
          "phone":       "+91 ****1234",
          "age_minutes": 12,
          "status":      "active" | "processing"
        },
        ...
      ]
    }
    """
    now = time.time()
    result = []

    for phone, session in list(user_sessions.items()):
        # Expire check (belt-and-suspenders)
        age_secs = now - session.get("timestamp", 0)
        if age_secs > SESSION_TTL_SECONDS:
            continue

        age_minutes = math.floor(age_secs / 60)

        # Mask the phone number  e.g.  +919876541234  →  +91 ****1234
        digits = "".join(filter(str.isdigit, phone))
        if len(digits) >= 10:
            last4    = digits[-4:]
            country  = "+" + digits[:-10] if len(digits) > 10 else ""
            masked   = f"{country} ****{last4}".strip()
        else:
            masked = "+** ****" + digits[-4:] if len(digits) >= 4 else "****"

        # Determine status
        if session.get("pending_replace"):
            status = "processing"
        else:
            status = "active"

        result.append({
            "phone":       masked,
            "age_minutes": age_minutes,
            "status":      status,
        })

    # Most-recent sessions first
    result.sort(key=lambda x: x["age_minutes"])

    return JSONResponse({
        "total":    len(result),
        "sessions": result,
    })


@app.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    log_section("INCOMING WEBHOOK")

    form         = await request.form()
    from_number  = form.get("From", "")
    to_number    = form.get("To",   "")
    incoming_msg = form.get("Body", "").strip()

    if incoming_msg.lower() == "reset":
        log_warning(f"Manual reset triggered for {from_number}")

        # Remove session safely
        user_sessions.pop(from_number, None)

        response = MessagingResponse()
        response.message(
            "🔄 Your session has been *reset* successfully.\n\n"
            "📄 Please upload a new medical report to continue."
        )

        return Response(str(response), media_type="text/xml")

    num_media    = int(form.get("NumMedia", 0))

    log_step(f"From     : {from_number}")
    log_step(f"To       : {to_number}")
    log_step(f"NumMedia : {num_media}")
    log_step(f"Body     : {incoming_msg!r}")

    response       = MessagingResponse()
    has_session    = _is_session_alive(from_number)
    session        = user_sessions.get(from_number, {})
    pending_replace= session.get("pending_replace", False)

    # ══════════════════════════════════════════════════════════════
    # BRANCH A: User is uploading a NEW report (media detected)
    # ══════════════════════════════════════════════════════════════
    if num_media > 0:
        log_step("BRANCH A: Media upload detected.")

        media_url    = form.get("MediaUrl0")
        content_type = form.get("MediaContentType0")
        log_step(f"MediaUrl0         : {media_url}")
        log_step(f"MediaContentType0 : {content_type}")

        # Step 3 (new report confirm): User already has an active session
        if has_session:
            log_warning("User already has an active report session. Asking for confirmation.")

            # Download the file first, then hold it pending confirmation
            file_name = download_media(media_url, content_type)
            if not file_name:
                response.message("❌ Failed to download your file. Please try again.")
                return Response(content=str(response), media_type="application/xml")

            file_path = os.path.join(UPLOAD_DIR, file_name)

            # Store pending state in session
            user_sessions[from_number]["pending_replace"]   = True
            user_sessions[from_number]["pending_file_path"] = file_path
            log_step(f"Pending replacement file queued: {file_path}")

            response.message(
                "⚠️ You already have an active report loaded.\n\n"
                "Uploading a new report will *replace* your current one and all previous "
                "context will be lost.\n\n"
                "Reply *YES* to confirm, or *NO* to keep your current report."
            )
            return Response(content=str(response), media_type="application/xml")

        # No existing session — proceed normally
        file_name = download_media(media_url, content_type)
        if not file_name:
            response.message("❌ Failed to download your file. Please try again.")
            return Response(content=str(response), media_type="application/xml")

        file_path = os.path.join(UPLOAD_DIR, file_name)
        log_success(f"Media downloaded → {file_path}")

        background_tasks.add_task(process_report_task, file_path, from_number, to_number)
        log_success("Background task enqueued.")

        response.message("⏳ Analyzing your medical report. Please hold on for a moment...")
        return Response(content=str(response), media_type="application/xml")

    # ══════════════════════════════════════════════════════════════
    # BRANCH B: User replied to a pending new-report confirmation
    # ══════════════════════════════════════════════════════════════
    elif pending_replace and incoming_msg:
        answer = incoming_msg.strip().upper()
        log_step(f"BRANCH B: Pending replace confirmation. Answer='{answer}'")

        if answer in ("YES", "Y", "CONFIRM"):
            queued_path = session.get("pending_file_path")
            log_step(f"User confirmed replacement. Processing: {queued_path}")

            # Clear old session first, then kick off new analysis
            del user_sessions[from_number]

            background_tasks.add_task(process_report_task, queued_path, from_number, to_number)
            response.message(
                "✅ Got it! Processing your new report now...\n"
                "⏳ Please hold on for a moment."
            )

        elif answer in ("NO", "N", "CANCEL"):
            _clear_pending(from_number)
            log_step("User cancelled replacement — keeping existing session.")
            response.message(
                "👍 Kept your original report. You can continue asking questions about it.\n"
                "_Example: 'What does my cholesterol level mean?'_"
            )

        else:
            # Unrecognised — re-prompt
            response.message(
                "Please reply *YES* to replace your report, or *NO* to keep your current one."
            )

        return Response(content=str(response), media_type="application/xml")

    # ══════════════════════════════════════════════════════════════
    # BRANCH C: Text message — user HAS an active report session
    # ══════════════════════════════════════════════════════════════
    elif has_session and incoming_msg:
        log_step("BRANCH C: Follow-up question with active session.")
        report_context = session.get("report_context", {})
        reply = chat_with_report_context(incoming_msg, report_context)
        response.message(reply)
        log_success("Follow-up response sent.")
        return Response(content=str(response), media_type="application/xml")

    # ══════════════════════════════════════════════════════════════
    # BRANCH D: Text message — NO active report (gatekeeper mode)
    # ══════════════════════════════════════════════════════════════
    elif incoming_msg:
        # Safety net: if somehow has_session was True but branch C was skipped, catch it here
        if _is_session_alive(from_number):
            log_failure("ROUTING BUG: Branch D reached but session IS alive — re-routing to C.")
            report_context = user_sessions.get(from_number, {}).get("report_context", {})
            reply = chat_with_report_context(incoming_msg, report_context)
            response.message(reply)
            log_success("Re-routed to follow-up mode. Follow-up response sent.")
            return Response(content=str(response), media_type="application/xml")

        log_step("BRANCH D: Text-only, no active session — gatekeeper mode.")
        reply = chat_with_gemini(incoming_msg)
        response.message(reply)
        log_success("Gatekeeper chat response sent.")
        return Response(content=str(response), media_type="application/xml")

    # ══════════════════════════════════════════════════════════════
    # BRANCH E: Empty / unrecognised
    # ══════════════════════════════════════════════════════════════
    else:
        log_warning("BRANCH E: Empty / unrecognised message.")
        response.message("📄 Please send a medical report (Image/PDF) to analyze.")
        return Response(content=str(response), media_type="application/xml")

# from fastapi import Request
# from fastapi.responses import Response
# from twilio.twiml.messaging_response import MessagingResponse

# @app.post("/webhook")
# async def whatsapp_webhook(request: Request):
#     response = MessagingResponse()
#     response.message("Hello from bot ✅")
#     print(str(response))
#     return Response(str(response), media_type="text/xml")

@app.get("/logs/stream")
async def stream_logs():
    """Server-Sent Events stream of real backend logs."""
    async def event_generator():
        last_index = max(0, len(log_buffer) - 20)  # send last 20 on connect
        while True:
            current_len = len(log_buffer)
            if current_len > last_index:
                for entry in list(log_buffer)[last_index:current_len]:
                    import json
                    yield f"data: {json.dumps(entry)}\n\n"
                last_index = current_len
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # important for nginx
        }
    )

