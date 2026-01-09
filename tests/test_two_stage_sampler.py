import json
import pytest

from seqsampling.models.base import ChatMessage, Generation, DummyClient
from seqsampling.sampling.config import SamplingConfig, GenerationConfig, ConstrainedGenConfig
from seqsampling.sampling.enumeration import EnumerationSampler, JsonSolutionsParser
from seqsampling.sampling.parallel import ParallelSampler, JsonSolutionParser
from seqsampling.sampling.two_stage import SeedPromptTemplate, TwoStageSampler


class SeedingPromptSchema:
    def build_messages(self, ctx):
        return [
            ChatMessage(role="system", content="seed system"),
            ChatMessage(role="user", content=f"input={ctx.input_text}"),
            ChatMessage(role="user", content=f"K={ctx.k}"),
        ]


class ExpansionPromptSchema:
    def build_messages(self, ctx):
        return [
            ChatMessage(role="system", content="expand system"),
            ChatMessage(role="user", content=f"input={ctx.input_text}"),
        ]


class SeedingDummyClient(DummyClient):
    def generate(
        self,
        messages,
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params=None,
    ):
        input_line = next(m.content for m in messages if m.content.startswith("input="))
        input_val = input_line.split("input=")[1]
        k_line = next(m.content for m in messages if m.content.startswith("K="))
        k = int(k_line.split("K=")[1])

        solutions = [f"{input_val}_seed{i}" for i in range(k)]
        text = json.dumps({"solutions": solutions})

        return [
            Generation(text=text, raw={"input": input_val, "n": idx})
            for idx in range(n)
        ]


class ExpansionDummyClient(DummyClient):
    def generate(
        self,
        messages,
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        extra_params=None,
    ):
        input_line = next(m.content for m in messages if m.content.startswith("input="))
        prompt_val = input_line.split("input=")[1]

        generations = []
        for idx in range(n):
            text = json.dumps({"solution": [f"expanded::{prompt_val}::{idx}"]})
            generations.append(Generation(text=text, raw={"prompt": prompt_val, "n": idx}))
        return generations


def test_seed_prompt_template_formats_seed_and_input():
    template = SeedPromptTemplate(template="Input:{input}|Seed{seed_index}:{seed}")
    prompt = template.render("task", {"topic": "x"}, 3)

    assert "Input:task" in prompt
    assert "Seed3" in prompt
    assert '"topic": "x"' in prompt


def test_two_stage_sampler_runs_pipeline_and_groups_results():
    seeding_sampler = EnumerationSampler(
        client=SeedingDummyClient(),
        prompt_schema=SeedingPromptSchema(),
        parser=JsonSolutionsParser(key="solutions"),
        sampling_config=SamplingConfig(
            k=2, generation_config=GenerationConfig(n=1, max_tokens=256)
        ),
        constrained_config=ConstrainedGenConfig(json_mode=False),
    )

    expansion_sampler = ParallelSampler(
        client=ExpansionDummyClient(),
        prompt_schema=ExpansionPromptSchema(),
        parser=JsonSolutionParser(key="solution"),
        sampling_config=SamplingConfig(
            k=1, generation_config=GenerationConfig(n=2, max_tokens=128)
        ),
        constrained_config=ConstrainedGenConfig(json_mode=False),
    )

    template = SeedPromptTemplate(template="Base:{input} | Seed {seed_index}:{seed}")
    sampler = TwoStageSampler(
        seeding_sampler=seeding_sampler,
        expansion_sampler=expansion_sampler,
        prompt_template=template,
    )

    inputs = ["task-one", "task-two"]
    results = sampler.run(inputs)

    assert len(results) == 2

    first = results[0]
    assert first.seeding_result.solutions_per_prompt == ["task-one_seed0", "task-one_seed1"]
    assert len(first.expansions) == 2
    assert first.expansions[0].templated_prompt.startswith("Base:task-one")
    assert len(first.expansions[0].parallel_result.solutions_per_prompt) == 2
    assert all(
        sol.startswith("expanded::Base:task-one | Seed")
        for sol in first.expansions[0].parallel_result.solutions_per_prompt
    )

    second_prompts = [exp.templated_prompt for exp in results[1].expansions]
    assert any("task-two_seed0" in p for p in second_prompts)
    assert any("task-two_seed1" in p for p in second_prompts)


def test_two_stage_sampler_requires_parallel_k_equal_one():
    seeding_sampler = EnumerationSampler(
        client=SeedingDummyClient(),
        prompt_schema=SeedingPromptSchema(),
        parser=JsonSolutionsParser(key="solutions"),
        sampling_config=SamplingConfig(
            k=1, generation_config=GenerationConfig(n=1, max_tokens=64)
        ),
        constrained_config=ConstrainedGenConfig(json_mode=False),
    )

    bad_expansion_sampler = ParallelSampler(
        client=ExpansionDummyClient(),
        prompt_schema=ExpansionPromptSchema(),
        parser=JsonSolutionParser(key="solution"),
        sampling_config=SamplingConfig(
            k=2, generation_config=GenerationConfig(n=1, max_tokens=32)
        ),
        constrained_config=ConstrainedGenConfig(json_mode=False),
    )

    sampler = TwoStageSampler(
        seeding_sampler=seeding_sampler,
        expansion_sampler=bad_expansion_sampler,
        prompt_template=SeedPromptTemplate(),
    )

    with pytest.raises(ValueError):
        sampler.run(["one task"])
