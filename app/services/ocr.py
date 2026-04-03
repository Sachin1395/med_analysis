import os
import json
import logging
import fitz
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2
from google import genai
from dotenv import load_dotenv
load_dotenv()
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # 🔥 DO NOT hardcode
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY as environment variable")

client = genai.Client(api_key=GEMINI_API_KEY)

POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
import re

def extract_json(text: str):
    try:
        return json.loads(text)
    except:
        pass

    # Try to extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — OCR (EYES)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image(img: Image.Image):
    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return Image.fromarray(binary)


def get_text_from_pdf_or_image(file_path: str) -> str:
    full_text = ""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)

        for page in doc:
            full_text += page.get_text()

        if len(full_text.strip()) < 100:
            log.info("PDF appears scanned → using OCR")
            images = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)

            for img in images:
                processed = preprocess_image(img)
                full_text += pytesseract.image_to_string(processed)

        doc.close()

    else:
        img = Image.open(file_path)
        full_text = pytesseract.image_to_string(preprocess_image(img))

    return full_text.strip()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — GEMINI (BRAIN)
# ═══════════════════════════════════════════════════════════════════════════════

def map_data_with_gemini(raw_text: str):

    print("\n--- RAW TEXT SAMPLE (DEBUG) ---\n")
    print(raw_text[:800])
    print("\n------------------------------\n")

    prompt = f"""
    Convert this medical report OCR text into STRICT JSON.

    Output ONLY valid JSON. No explanation. No markdown.

    Format:
    {{
      "patient_info": {{
        "name": "",
        "id": "",
        "age": "",
        "gender": "",
        "report_date": ""
      }},
      "test_results": [
        {{
          "test": "",
          "value": "",
          "unit": "",
          "range": "",
          "status": ""
        }}
      ]
    }}

    TEXT:
    {raw_text}
    """

    log.info("Sending data to Gemini...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    cleaned = extract_json(response.text)

    if not cleaned:
        return {
            "error": "Invalid JSON from Gemini",
            "raw_response": response.text[:1000]
        }

    return cleaned

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_medical_report(file_path: str):

    raw_text = get_text_from_pdf_or_image(file_path)

    if not raw_text:
        return {"error": "No text extracted"}

    try:
        return map_data_with_gemini(raw_text)

    except Exception as e:
        log.error(f"Gemini failed: {e}")
        return {
            "error": str(e),
            "raw_text": raw_text[:1000]
        }

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    file_path = r"..\..\downloads\Investigation.pdf"

    result = process_medical_report(file_path)

    print("\n" + "═" * 60)
    print("  MEDICAL REPORT EXTRACTION")
    print("═" * 60)

    if "error" in result:
        print("❌ ERROR:", result["error"])
        exit()

    patient = result.get("patient_info", {})

    print(f"NAME: {patient.get('name')}")
    print(f"ID: {patient.get('id')}")
    print(f"AGE/GENDER: {patient.get('age')} / {patient.get('gender')}")
    print("-" * 60)

    print(f"{'Test':<30} {'Value':<10} {'Unit':<10} {'Status'}")
    print("-" * 60)

    for t in result.get("test_results", []):
        print(f"{t.get('test','')[:28]:<30} {t.get('value',''):<10} {t.get('unit',''):<10} {t.get('status','')}")

    with open("Extraction_Cleaned.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n✅ Saved → Extraction_Cleaned.json")