# src/seqsampling/models/vllm_client.py
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx  # add to your dependencies

from .base import ChatMessage, Generation, LLMClient, LLMError, to_openai_chat_format


@dataclass
class VLLMClient(LLMClient):
    """
    Client for a vLLM OpenAI-compatible server.

    This client assumes the vLLM server exposes a /v1/chat/completions endpoint
    with OpenAI-like semantics, e.g.:

        POST {base_url}/v1/chat/completions
        {
          "model": "your-model-name",
          "messages": [...],
          "n": 2,
          "max_tokens": 128,
          ...
        }

    Parameters
    ----------
    base_url:
        Base URL of the vLLM server, e.g. "http://localhost:8000".
        Do not include the path; the client will append "/v1/chat/completions".
    model:
        Name/identifier of the model, as configured in the vLLM server.
    api_key:
        Optional API key; if set, it will be sent as a Bearer token in the
        Authorization header (useful if you apply simple auth in front of vLLM).
    timeout:
        Request timeout in seconds.
    default_headers:
        Additional headers to send with every request.
    """

    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout: float = 300.0
    default_headers: Dict[str, str] = field(default_factory=dict)
    max_concurrent: int = 16        # limit for batched async calls

    # internal reusable clients
    _client: Optional[httpx.Client] = field(default=None, init=False, repr=False)
    _aclient: Optional[httpx.AsyncClient] = field(default=None, init=False, repr=False)

    # ---------- public API ----------

    def generate(
        self,
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

        client = self._get_client()
        try:
            response = client.post(
                self._chat_url,
                json=payload,
                timeout=self.timeout,
                headers=self._build_headers(),
            )
        except httpx.RequestError as e:
            raise LLMError(f"vLLM request failed: {e!r}") from e

        self._raise_for_status(response)

        data = self._parse_json(response)
        return self._extract_generations(data)

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
        async with httpx.AsyncClient(timeout=self.timeout) as aclient:
            return await self._agenerate_with_client(
                aclient=aclient,
                messages=messages,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_params=extra_params,
            )

    async def _agenerate_with_client(
        self,
        aclient: httpx.AsyncClient,
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
            response = await aclient.post(
                self._chat_url,
                json=payload,
                timeout=self.timeout,
                headers=self._build_headers(),
            )
        except httpx.RequestError as e:
            raise LLMError(f"vLLM async request failed: {e!r}") from e

        self._raise_for_status(response)

        data = self._parse_json(response)
        return self._extract_generations(data)
    
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

            async with httpx.AsyncClient(timeout=self.timeout) as aclient:
                async def _one(messages: List[ChatMessage]) -> List[Generation]:
                    async with sem:
                        return await self._agenerate_with_client(
                            aclient=aclient,
                            messages=messages,
                            n=n,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            extra_params=extra_params,
                        )

                tasks = [asyncio.create_task(_one(msgs)) for msgs in batch_messages]
                return await asyncio.gather(*tasks)

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

    def close(self) -> None:
        """
        Close underlying HTTP clients.

        Not strictly required (httpx will clean up on GC),
        but good practice in long-running processes.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._aclient is not None:
            try:
                # In non-async contexts, this is best-effort;
                # user can close manually in their own loop as well.
                import anyio

                anyio.run(self._aclient.aclose)
            except Exception:
                # Fallback: ignore errors on background close.
                pass
            self._aclient = None

    # ---------- internal helpers ----------

    @property
    def _chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self.timeout)
        return self._aclient

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.default_headers)
        return headers

    def _build_payload(
        self,
        messages: List[ChatMessage],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_params: Optional[Dict[str, Any]] = None,
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
            # extra_params can be used for:
            # - stop
            # - presence_penalty / frequency_penalty
            # - "guided_json", "grammar", etc. depending on vLLM config
            payload.update(extra_params)
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                data = response.json()
                message = data.get("error", {}).get("message")
            except Exception:
                message = None
            msg = f"vLLM server returned HTTP {response.status_code}"
            if message:
                msg += f": {message}"
            raise LLMError(msg)

    @staticmethod
    def _parse_json(response: httpx.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON from vLLM server: {e}") from e

    @staticmethod
    def _extract_generations(data: Dict[str, Any]) -> List[Generation]:
        """
        Extract Generation objects from an OpenAI-style chat completion payload.
        Expected shape:

            {
              "id": "...",
              "object": "chat.completion",
              "choices": [
                {
                  "index": 0,
                  "message": {"role": "assistant", "content": "..."},
                  "finish_reason": "stop"
                },
                ...
              ],
              "usage": {...}
            }
        """
        choices = data.get("choices")
        if not isinstance(choices, list):
            raise LLMError("vLLM response missing 'choices' list.")

        generations: List[Generation] = []
        for choice in choices:
            msg = choice.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, str):
                raise LLMError("vLLM choice has no 'message.content' string.")
            generations.append(Generation(text=content, raw=choice))

        return generations
