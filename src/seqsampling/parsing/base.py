# src/seqsampling/parsing/base.py
from typing import Any, List, Protocol

class ParseError(Exception):
    pass

PARSE_FAILED_PLACEHOLDER = "<PARSE_FAILED>"

class Parser(Protocol):
    """Turn model output string into a list of 'solutions'."""

    def parse(self, text: str) -> List[Any]:
        ...

class ExtractionParser(Protocol):
    """Extract a specific part from the model output string."""

    def extract(self, text: str) -> str:
        ...