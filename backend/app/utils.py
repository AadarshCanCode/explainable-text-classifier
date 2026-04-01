import re

def clean_text(text: str) -> str:
    """Basic text cleaning utility."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits (optional, depends on use case)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
