

def safety_node(state: dict):

    patterns = state.get("patterns", [])
    explanations = state.get("explanation", {})

    safe_patterns = []
    safe_explanations = {}

    def make_safe(text):
        text = text.strip()

        # Avoid direct diagnosis
        replacements = [
            ("indicates", "may suggest"),
            ("confirms", "may indicate"),
            ("diagnosis", "possible condition"),
            ("is", "may be")
        ]

        for old, new in replacements:
            text = text.replace(old, new)

        # Ensure safe prefix
        if not text.lower().startswith(("may", "possible")):
            text = "May suggest " + text

        return text

    for p in patterns:
        safe_patterns.append(make_safe(p))

    for k, v in explanations.items():
        safe_explanations[k] = make_safe(v)

    return {
        **state,
        "patterns": safe_patterns,
        "explanation": safe_explanations
    }