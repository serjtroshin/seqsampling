# src/seqsampling/models/openai_client.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI, AsyncOpenAI

from .base import ChatMessage, Generation, LLMClient, LLMError, to_openai_chat_format


@dataclass
class OpenAIClient(LLMClient):
    """
    Wrapper for the official OpenAI client using the chat.completions API.

    Features:
    - `generate(...)`: single prompt, sync.
    - `generate_batch(...)`: multiple prompts, sync API, async under the hood.
    """

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # optional for Azure or custom gateways
    timeout: float = 120.0
    max_concurrent: int = 16        # limit for batched async calls

    def __post_init__(self) -> None:
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        messages: List[ChatMessage],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_chat_format(messages),
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if extra_params:
            payload.update(extra_params)
        return payload

    @staticmethod
    def _extract_generations(response) -> List[Generation]:
        gens: List[Generation] = []
        for choice in response.choices:
            text = choice.message.content
            if text is None:
                raise LLMError("OpenAI returned a null message content.")
            gens.append(Generation(text=text, raw=choice.model_dump()))
        return gens

    def _build_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------ #
    # Single-request sync generate (existing API)
    # ------------------------------------------------------------------ #

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
        Synchronous generate for a single prompt.
        """
        payload = self._build_payload(
            messages=messages,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_params=extra_params,
        )

        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as e:
            raise LLMError(f"OpenAI request failed: {e}") from e

        return self._extract_generations(response)

    # ------------------------------------------------------------------ #
    # Single-request async generate (internal / advanced use)
    # ------------------------------------------------------------------ #

    async def _agenerate_with_client(
        self,
        async_client: AsyncOpenAI,
        messages: List[ChatMessage],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Generation]:
        payload = self._build_payload(
            messages=messages,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_params=extra_params,
        )

        try:
            response = await async_client.chat.completions.create(**payload)
        except Exception as e:
            raise LLMError(f"OpenAI async request failed: {e}") from e

        return self._extract_generations(response)

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
        Asynchronous generate for a single prompt.
        """
        async_client = self._build_async_client()
        try:
            return await self._agenerate_with_client(
                async_client=async_client,
                messages=messages,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_params=extra_params,
            )
        finally:
            try:
                await async_client.close()
            except Exception:
                # Ignore close errors so upstream exceptions propagate cleanly.
                pass

    # ------------------------------------------------------------------ #
    # Batched sync generate (what you want to use by default)
    # ------------------------------------------------------------------ #

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
        Synchronous batched generate.

        Parameters
        ----------
        batch_messages:
            List of message lists, one per prompt.
        n, max_tokens, temperature, top_p, extra_params:
            Same semantics as in `generate`.

        Returns
        -------
        List[List[Generation]]
            Outer list: one entry per prompt.
            Inner list: generations for that prompt.
        """

        async def _run_batch():
            sem = asyncio.Semaphore(self.max_concurrent)
            # Fresh AsyncOpenAI per batch avoids reusing a client tied to a
            # closed event loop when asyncio.run spins up new loops.
            async_client = self._build_async_client()

            async def _one(messages: List[ChatMessage]) -> List[Generation]:
                async with sem:
                    return await self._agenerate_with_client(
                        async_client=async_client,
                        messages=messages,
                        n=n,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        extra_params=extra_params,
                    )

            try:
                tasks = [asyncio.create_task(_one(msgs)) for msgs in batch_messages]
                return await asyncio.gather(*tasks)
            finally:
                try:
                    await async_client.close()
                except Exception:
                    pass

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # no running event loop → safe to use asyncio.run
            return asyncio.run(_run_batch())
        else:
            # already inside an event loop (e.g. Jupyter) → run in that loop
            future = asyncio.run_coroutine_threadsafe(_run_batch(), loop)
            return future.result()
