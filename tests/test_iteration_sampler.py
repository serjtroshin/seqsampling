import json
from typing import Any, Dict, List, Optional

from seqsampling.models.base import ChatMessage, Generation, LLMClient, DummyClient
from seqsampling.sampling.config import SamplingConfig, GenerationConfig, ConstrainedGenConfig
from seqsampling.sampling.iteration import IterationSampler, JsonPromptSchema, JsonSolutionsParser


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
        ]


class DummyClient(DummyClient):
    """
    Deterministic client for tests.

    Reads the last message "K={k}" and returns a JSON string with exactly k solutions.
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
        # Grab k from the last message

        solutions = [f"solution"]
        text = json.dumps({"solutions": solutions})

        return [Generation(text=text, raw={"mock": True}) for _ in range(n)]  # generate n parallel solutions


def test_sequential_sampler_shapes():
    n = 2
    k = 3

    client = DummyClient()
    prompt_schema = DummyPromptSchema()
    parser = JsonSolutionsParser(key="solutions")

    sampling_cfg = SamplingConfig(k=k, generation_config=GenerationConfig(n=n, max_tokens=512, temperature=0.9))
    constrained_cfg = ConstrainedGenConfig(json_mode=False)

    sampler = IterationSampler(
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
        assert sols.startswith("solution")
