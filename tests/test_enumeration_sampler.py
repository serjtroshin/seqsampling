from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional

from seqsampling.models.base import ChatMessage, Generation, LLMClient, DummyClient
from seqsampling.parsing.base import ParseError
from seqsampling.sampling.config import SamplingConfig, GenerationConfig, ConstrainedGenConfig
from seqsampling.sampling.enumeration import EnumerationSampler, JsonPromptSchema, JsonSolutionsParser


class DummyPromptSchema:
    """
    Minimal prompt schema for testing.

    We only care that `ctx.k` is encoded somehow so DummyClient can read it.
    """

    def build_messages(self, ctx) -> List[ChatMessage]:
        # ctx has attributes: input_text, k (as in your real PromptContext)
        return [
            ChatMessage(role="system", content="dummy system"),
            ChatMessage(role="user", content=f"input={ctx.input_text}"),
            ChatMessage(role="user", content=f"K={ctx.k}"),
        ]


class EnumerationDummyClient(DummyClient):
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
        # Extract k from the last message: expects "K={k}"
        last = messages[-1].content
        k = int(last.split("K=")[1])

        solutions = [f"solution_{i}" for i in range(k)]
        text = json.dumps({"solutions": solutions})

        # Return n parallel generations (same JSON each time)
        return [
            Generation(text=text, raw={"mock": True, "n": n})
            for _ in range(n)
        ]

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


def test_json_solutions_parser_happy_path():
    text = """
    {
      "solutions": [
        {"answer": "A", "score": 0.9},
        {"answer": "B", "score": 0.7},
        {"answer": "C", "score": 0.4}
      ]
    }
    """
    parser = JsonSolutionsParser(key="solutions")
    solutions = parser.parse(text)

    assert isinstance(solutions, list)
    assert len(solutions) == 3
    assert solutions[0]["answer"] == "A"

def test_json_solutions_parser_raises_on_invalid_json():
    text = "{ not valid json }"
    parser = JsonSolutionsParser()

    try:
        parser.parse(text)
        assert False, "ParseError expected"
    except ParseError:
        pass

def test_sequential_sampler_shapes():
    n = 2
    k = 3

    client = EnumerationDummyClient()
    prompt_schema = DummyPromptSchema()
    parser = JsonSolutionsParser(key="solutions")

    sampling_cfg = SamplingConfig(k=k, generation_config=GenerationConfig(n=n, max_tokens=512, temperature=0.9))
    constrained_cfg = ConstrainedGenConfig(json_mode=False)

    sampler = EnumerationSampler(
        client=client,
        prompt_schema=prompt_schema,
        parser=parser,
        sampling_config=sampling_cfg,
        constrained_config=constrained_cfg,
    )

    inputs = ["task one", "task two"]
    results = sampler.run(inputs)

    assert len(results[0].solutions_per_prompt) == n * k
    for sols in results[0].solutions_per_prompt:
        assert sols.startswith("solution_")
