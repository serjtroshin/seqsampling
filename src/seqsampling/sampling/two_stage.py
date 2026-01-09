from dataclasses import dataclass
import json
from typing import Any, List, Tuple

from seqsampling.sampling.sampler import SequentialSampler, SequentialSamplingResult


@dataclass
class SeedPromptTemplate:
    """
    Simple formatter that injects the original input and a seed into a prompt
    template. Available fields: {input}, {seed}, {seed_index}.
    """
    template: str = "Original input: {input}\nSeed {seed_index}: {seed}"

    def render(self, input_text: str, seed: Any, seed_index: int) -> str:
        return self.template.format(
            input=input_text,
            seed=self._seed_to_text(seed),
            seed_index=seed_index,
        )

    @staticmethod
    def _seed_to_text(seed: Any) -> str:
        if isinstance(seed, str):
            return seed
        try:
            return json.dumps(seed)
        except TypeError:
            return str(seed)


@dataclass
class SeedExpansion:
    """
    Container linking a single seed to the result of its parallel expansion.
    """
    seed: Any
    seed_index: int
    templated_prompt: str
    parallel_result: SequentialSamplingResult

    @property
    def solutions(self) -> List[Any]:
        return self.parallel_result.solutions_per_prompt


@dataclass
class TwoStageSamplingResult:
    """
    Results for a single original input: seeds from the first stage plus all
    parallel expansions produced from those seeds.
    """
    input_text: str
    seeding_result: SequentialSamplingResult
    expansions: List[SeedExpansion]

    @property
    def expanded_solutions(self) -> List[Any]:
        flattened: List[Any] = []
        for exp in self.expansions:
            flattened.extend(exp.parallel_result.solutions_per_prompt)
        return flattened


@dataclass
class TwoStageSampler:
    """
    Orchestrates a two-stage workflow:
      1) Run a SequentialSampler (e.g., EnumerationSampler or IterationSampler)
         to propose seeds.
      2) For each seed, build a templated prompt and run a ParallelSampler to
         expand it with multiple parallel generations.
    """
    seeding_sampler: SequentialSampler
    expansion_sampler: SequentialSampler  # expected to be ParallelSampler
    prompt_template: SeedPromptTemplate

    def run(self, inputs: List[str]) -> List[TwoStageSamplingResult]:
        seed_results = self.seeding_sampler.run(inputs)

        templated_prompts: List[str] = []
        prompt_index: List[Tuple[int, int, Any, str]] = []  # (input_idx, seed_idx, seed, prompt)
        for input_idx, seed_result in enumerate(seed_results):
            for seed_idx, seed in enumerate(seed_result.solutions_per_prompt):
                prompt = self.prompt_template.render(inputs[input_idx], seed, seed_idx)
                templated_prompts.append(prompt)
                prompt_index.append((input_idx, seed_idx, seed, prompt))

        if getattr(self.expansion_sampler, "sampling_config", None) is not None:
            if self.expansion_sampler.sampling_config.k != 1:
                raise ValueError(
                    "TwoStageSampler expects expansion_sampler.sampling_config.k == 1 "
                    "so the parallel stage can control fan-out via GenerationConfig.n."
                )

        if not templated_prompts:
            return [
                TwoStageSamplingResult(
                    input_text=inputs[idx],
                    seeding_result=seed_results[idx],
                    expansions=[],
                )
                for idx in range(len(inputs))
            ]

        expansion_results = self.expansion_sampler.run(templated_prompts)

        if len(expansion_results) != len(prompt_index):
            raise RuntimeError(
                f"Expected {len(prompt_index)} expansion results, got {len(expansion_results)}."
            )

        grouped_expansions: List[List[SeedExpansion]] = [[] for _ in inputs]
        for exp_res, (input_idx, seed_idx, seed, prompt) in zip(expansion_results, prompt_index):
            grouped_expansions[input_idx].append(
                SeedExpansion(
                    seed=seed,
                    seed_index=seed_idx,
                    templated_prompt=prompt,
                    parallel_result=exp_res,
                )
            )

        return [
            TwoStageSamplingResult(
                input_text=inputs[idx],
                seeding_result=seed_results[idx],
                expansions=grouped_expansions[idx],
            )
            for idx in range(len(inputs))
        ]
