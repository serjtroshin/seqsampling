import json
from typing import Any, Dict, List, Optional

from seqsampling.models.base import ChatMessage, Generation, DummyClient
from seqsampling.sampling.config import SamplingConfig, GenerationConfig, ConstrainedGenConfig
from seqsampling.sampling.parallel import ParallelSampler, JsonSolutionParser


class DummyPromptSchema:
    """
    Minimal prompt schema for testing parallel sampling.
    """

    def build_messages(self, ctx) -> List[ChatMessage]:
        return [
            ChatMessage(role="system", content="dummy system"),
            ChatMessage(role="user", content=f"input={ctx.input_text}"),
            ChatMessage(role="user", content=f"K={ctx.k}"),
        ]


class ParallelDummyClient(DummyClient):
    """
    Deterministic client that emits a single JSON-wrapped solution per generation.
    """

    def __init__(self):
        self.call_counter = 0

    def generate(
        self,
        messages: List[ChatMessage],
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Generation]:
        self.call_counter += 1
        input_line = next(m.content for m in messages if m.content.startswith("input="))
        input_val = input_line.split("input=")[1]

        generations: List[Generation] = []
        for idx in range(n):
            solution = f"{input_val}_call{self.call_counter}_n{idx}"
            text = json.dumps({"solution": [solution]})
            generations.append(Generation(text=text, raw={"call": self.call_counter, "n": idx}))
        return generations


def test_json_solution_parser_happy_path():
    text = """
    {
      "solution": [
        {"answer": "A", "score": 0.9}
      ]
    }
    """
    parser = JsonSolutionParser(key="solution")
    solution = parser.parse(text)

    assert isinstance(solution, dict)
    assert solution["answer"] == "A"


def test_parallel_sampler_shapes_and_ids():
    n = 3
    k = 1

    client = ParallelDummyClient()
    prompt_schema = DummyPromptSchema()
    parser = JsonSolutionParser(key="solution")

    sampling_cfg = SamplingConfig(k=k, generation_config=GenerationConfig(n=n, max_tokens=256, temperature=0.7))
    constrained_cfg = ConstrainedGenConfig(json_mode=False)

    sampler = ParallelSampler(
        client=client,
        prompt_schema=prompt_schema,
        parser=parser,
        sampling_config=sampling_cfg,
        constrained_config=constrained_cfg,
    )

    inputs = ["task one", "task two"]
    results = sampler.run(inputs)

    assert len(results) == len(inputs)

    expected_seq_ids = [0 for _ in range(n)]
    expected_par_ids = [pid for pid in range(n)]

    for res, prefix in zip(results, ["task one", "task two"]):
        assert len(res.solutions_per_prompt) == n * k
        assert len(res.raw_generations) == n * k
        assert res.sequential_ids == expected_seq_ids
        assert res.parallel_ids == expected_par_ids
        assert all(sol.startswith(prefix) for sol in res.solutions_per_prompt)
