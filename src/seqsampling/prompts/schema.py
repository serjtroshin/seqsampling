# src/seqsampling/prompts/schema.py
from dataclasses import dataclass
from typing import List, Protocol

from ..models.base import ChatMessage


@dataclass
class PromptContext:
    input_text: str
    k: int


class PromptSchema(Protocol):
    """How to build chat messages for a task."""
    def build_messages(self, ctx: PromptContext) -> List[ChatMessage]:
        ...

    def supports_iteration(self) -> bool:
        """Whether this schema supports iterative sampling."""
        return False