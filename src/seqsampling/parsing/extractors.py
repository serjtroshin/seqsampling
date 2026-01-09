from .base import ExtractionParser

class RemoveThinkingParser(ExtractionParser):
    def extract(self, text: str, end_del="</think>") -> str:
        """Extract the part of the text after </think> tag, if present."""
        end_tag = end_del.lower()
        lower_text = text.lower()
        end_idx = lower_text.find(end_tag)
        if end_idx != -1:
            return text[end_idx + len(end_tag):].strip()
        return text.strip()
        