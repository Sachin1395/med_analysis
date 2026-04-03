from app.services.ocr import process_medical_report

def ingest_node(state: dict):
    file_path = state.get("file_path")

    if not file_path:
        return {"error": "No file path provided"}

    try:
        report = process_medical_report(file_path)

        # 🔴 Handle failure case
        if "error" in report:
            return {
                "error": report["error"],
                "raw_text": report.get("raw_text", ""),
                "structured_report": {},
                "warnings": ["AI extraction failed"]
            }

        # ✅ Extract safely from dict
        patient_info = report.get("patient_info", {})
        test_results = report.get("test_results", [])

        structured = {
            "patient_info": patient_info,
            "test_results": [
                {
                    "test": r.get("test"),
                    "value": r.get("value"),
                    "unit": r.get("unit"),
                    "range": r.get("range"),
                    "status": r.get("status"),
                }
                for r in test_results
            ]
        }

        return {
            "raw_text": "",  # optional (you removed raw_text from OCR return)
            "structured_report": structured,
            "warnings": []
        }

    except Exception as e:
        return {
            "error": str(e),
            "structured_report": {},
            "warnings": ["Unexpected failure"]
        }