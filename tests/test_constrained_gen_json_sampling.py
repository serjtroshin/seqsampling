# tests/test_constrained_json_sampling.py
import json
from typing import Any, Dict, List, Optional

import pytest

from seqsampling.models.base import ChatMessage, Generation, LLMClient, DummyClient
from seqsampling.parsing.base import ParseError
from seqsampling.sampling.config import SamplingConfig, GenerationConfig, ConstrainedGenConfig
from seqsampling.sampling.enumeration import EnumerationSampler, JsonSolutionsParser, JsonPromptSchema


class DummyPromptSchema:
    """Minimal PromptSchema-like object."""

    def build_messages(self, ctx) -> List[ChatMessage]:
        # ctx has: input_text, k
        return [
            ChatMessage(role="system", content="dummy system"),
            ChatMessage(role="user", content=f"task={ctx.input_text}"),
        ]


class DummyConstrainedClient(DummyClient):
    """
    Dummy client that simulates a backend obeying response_format constraints.
    - If response_format.type == 'json_object', returns valid JSON.
    - Otherwise returns non-JSON text.
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
        extra_params = extra_params or {}
        resp_format = extra_params.get("response_format", {})
        is_json = resp_format.get("type") == "json_object"

        if is_json:
            # Produce valid JSON with a list of k dummy solutions
            # We don't know k here directly; keep it simple and always do 3
            payload = {"solutions": ["a", "b", "c"]}
            text = json.dumps(payload)
        else:
            # Bad output that the JSON parser will choke on
            text = "this is not json"

        return [Generation(text=text, raw={"extra_params": extra_params})]

    async def agenerate(self, *args, **kwargs):
        raise NotImplementedError


def make_sampler(n:int, k: int, json_mode: bool) -> EnumerationSampler:
    client = DummyConstrainedClient()
    prompt_schema = DummyPromptSchema()
    parser = JsonSolutionsParser(key="solutions")
    sampling_cfg = SamplingConfig(
        k=k,
        generation_config=GenerationConfig(n=n, max_tokens=64, temperature=0.0),
        extra_params={},
    )
    constrained_cfg = ConstrainedGenConfig(json_mode=json_mode)

    return EnumerationSampler(
        client=client,
        prompt_schema=prompt_schema,
        parser=parser,
        sampling_config=sampling_cfg,
        constrained_config=constrained_cfg,
    )


class BadJsonClient(DummyClient):
    """
    Returns invalid JSON regardless of requested format.
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
        return [Generation(text="not json", raw={"bad": True})]


def test_sequential_sampler_with_json_mode_succeeds():
    sampler = make_sampler(n=1, k=3, json_mode=True)
    inputs = ["dummy task"]
    results = sampler.run(inputs)

    assert len(results[0].solutions_per_prompt) == 3
    sols = results[0].solutions_per_prompt
    assert len(sols) == 3
    assert sols == ["a", "b", "c"]


def test_sequential_sampler_without_json_mode_fails():
    sampler = make_sampler(n=1, k=3, json_mode=False)
    inputs = ["dummy task"]

    with pytest.warns(UserWarning, match="Parsing failed for generation"):
        sampler.run(inputs)


def test_sequential_sampler_with_json_mode_raises_on_parse_failure():
    client = BadJsonClient()
    prompt_schema = DummyPromptSchema()
    parser = JsonSolutionsParser(key="solutions")
    sampling_cfg = SamplingConfig(
        k=1,
        generation_config=GenerationConfig(n=1, max_tokens=64, temperature=0.0),
        extra_params={},
    )
    constrained_cfg = ConstrainedGenConfig(json_mode=True)

    sampler = EnumerationSampler(
        client=client,
        prompt_schema=prompt_schema,
        parser=parser,
        sampling_config=sampling_cfg,
        constrained_config=constrained_cfg,
    )

    with pytest.raises(ParseError):
        sampler.run(["dummy task"])
