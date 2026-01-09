# src/seqsampling/models/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Literal


ChatRole = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """
    Backend-agnostic representation of a chat message.
    """
    role: ChatRole
    content: str


@dataclass
class Generation:
    """
    A single model generation, with the text + raw backend payload.
    """
    text: str
    raw: Dict[str, Any]


class LLMError(RuntimeError):
    """
    Generic error type for LLM backends.
    """
    pass


@runtime_checkable
class LLMClient(Protocol):
    """
    Backend-agnostic interface for chat-template language models.

    Implementations wrap vLLM, transformers, OpenAI-compatible APIs, etc.
    """

    def generate(
        self,
        messages: List[ChatMessage],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Generation]:
        """
        Run a (non-streaming) chat completion call.

        Parameters
        ----------
        messages:
            Conversation so far, as a list of ChatMessage objects.
        n:
            Number of independent generations to sample (e.g., OpenAI 'n').
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature.
        top_p:
            Nucleus sampling parameter.
        extra_params:
            Backend-specific additional parameters (e.g. stop, penalties,
            grammar constraints, etc.).

        Returns
        -------
        List[Generation]
            One Generation per sampled completion.
        """
        ...

    async def agenerate(
        self,
        messages: List[ChatMessage],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Generation]:
        """
        Async variant of `generate`. Implementations may just wrap the sync
        version in a thread pool if true async is not available.
        """
        ...

    def generate_batch(
        self,
        batch_messages: List[List[ChatMessage]],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Generation]]:
        """
        Batched variant of `generate` for multiple prompts.
        Run a (non-streaming) chat completion call.

        Parameters
        ----------
        messages:
            Conversation so far, as a list of ChatMessage objects.
        n:
            Number of independent generations to sample (e.g., OpenAI 'n').
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature.
        top_p:
            Nucleus sampling parameter.
        extra_params:
            Backend-specific additional parameters (e.g. stop, penalties,
            grammar constraints, etc.).

        Returns
        -------
        List[List[Generation]]
            batch of generations, one list per prompt.
        """
        ...


def to_openai_chat_format(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """
    Convert internal ChatMessage objects to OpenAI/vLLM-style dicts.

    Example output item: {"role": "user", "content": "Hello"}
    """
    return [{"role": m.role, "content": m.content} for m in messages]



class DummyClient(LLMClient):
    """
    Deterministic client for tests.

    Reads the last message "K={k}" and returns exactly k solutions:
        {"solutions": ["solution_0", ..., "solution_{k-1}"]}

    Supports:
      - generate(...)       -> single prompt
      - agenerate(...)      -> async wrapper around generate
      - generate_batch(...) -> multiple prompts (sync), used to test batching
    """

    # ------------------ single-request sync ------------------ #

    def generate(
        self,
        messages: List[ChatMessage],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Generation]:
        ...

    # ------------------ single-request async ------------------ #

    async def agenerate(
        self,
        messages: List[ChatMessage],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Generation]:
        """
        Async wrapper around generate() so async code paths can be tested.
        """
        await asyncio.sleep(0)  # yield to event loop
        return self.generate(
            messages=messages,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_params=extra_params,
        )

    # ------------------ batched sync API ------------------ #

    def generate_batch(
        self,
        batch_messages: List[List[ChatMessage]],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Generation]]:
        """
        Batched generate for tests, mirroring OpenAIClient.generate_batch.

        Parameters
        ----------
        batch_messages:
            List of message lists; one per prompt.

        Returns
        -------
        List[List[Generation]]
            Outer list: one entry per prompt.
            Inner list: list of Generation objects for that prompt.
        """
        return [
            self.generate(
                messages=messages,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_params=extra_params,
            )
            for messages in batch_messages
        ]