from dataclasses import asdict, dataclass
import json
from typing import Any, List, Optional, Type

from pydantic import BaseModel, ValidationError

from seqsampling.parsing.base import Parser, ParseError, PARSE_FAILED_PLACEHOLDER
from seqsampling.prompts.schema import PromptSchema, PromptContext, ChatMessage
from seqsampling.sampling.sampler import SequentialSampler, SequentialSamplingResult


@dataclass
class JsonPromptSchema(PromptSchema):
    """
    Simple schema asking for a single JSON-wrapped solution.
    """
    system_instruction: str
    field_name: str = "solution"  # configurable via config
    example_json: str = '{"solution": ["<Solution>. The answer is <Answer>."]}'  # example for clarity

    def build_messages(self, ctx: PromptContext) -> List[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    "You must output a JSON object with a field "
                    f"'{self.field_name}', which is an array of exactly "
                    f"{1} item. Example: {self.example_json}\n\n"
                    f"Task input:\n{ctx.input_text}"
                ),
            ),
        ]


class JsonSolutionParser(Parser):
    def __init__(
        self,
        key: str = "solution",
        item_model: Optional[Type[BaseModel]] = None,
    ):
        self.key = key
        self.item_model = item_model

    def parse(self, text: str) -> Any:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}") from e

        if self.key not in obj:
            raise ParseError(f"Missing key '{self.key}' in JSON.")

        item = obj[self.key]
        if not isinstance(item, list):
            raise ParseError(f"'{self.key}' should be a list.")
        if len(item) != 1:
            raise ParseError(f"Expected exactly 1 item under '{self.key}', got {len(item)}.")
        item = item[0]

        if self.item_model is None:
            return item

        try:
            parsed_item = self.item_model.model_validate(item)
        except ValidationError as e:
            raise ParseError(f"Item failed validation: {e}") from e
        return parsed_item


@dataclass
class ParallelSampler(SequentialSampler):
    """
    Sampler that issues independent generations for each input in parallel.
    For this sampler, we enforce k=1 and rely on GenerationConfig.n to control
    the number of parallel samples.
    """
    def __post_init__(self):
        if self.prompt_schema is None:
            self.prompt_schema = JsonPromptSchema(system_instruction=self.system_instruction)
        if self.parser is None:
            self.parser = JsonSolutionParser(key=self.prompt_schema.field_name)

    def run(self, inputs: List[str]) -> List[SequentialSamplingResult]:
        """
        inputs: list of input texts
        For each input, launch a single model call (batched) and parse 1 solution
        from each of the `n` parallel generations.
        """
        if self.sampling_config.k != 1:
            raise ValueError(
                "ParallelSampler expects sampling_config.k == 1; "
                "use GenerationConfig.n to control parallel samples."
            )

        print(
            f"Running ParallelSampler on {len(inputs)} inputs with "
            f"n={self.sampling_config.generation_config.n}, k={self.sampling_config.k}"
        )

        raw_generations: List[List[str]] = [[] for _ in inputs]
        all_solutions: List[List[Any]] = [[] for _ in inputs]
        parallel_ids: List[List[int]] = [[] for _ in inputs]
        sequential_ids: List[List[int]] = [[] for _ in inputs]

        # Build a single message template per input.
        message_batches = [
            self.prompt_schema.build_messages(
                PromptContext(input_text=inp, k=self.sampling_config.k)
            )
            for inp in inputs
        ]

        extra = self.sampling_config.extra_params or {}
        extra = {**extra, **self._constrained_extra_params()}
        gen_cfg = self.sampling_config.generation_config

        batched_gens = self.client.generate_batch(
            batch_messages=message_batches,
            **asdict(gen_cfg),
            extra_params=extra,
        )

        for input_id, gens in enumerate(batched_gens):
            for par_id, gen in enumerate(gens):
                text = gen.text
                raw_generations[input_id].append(text)
                text = self._extract_answer(text)
                try:
                    parsed = self.parser.parse(text) if self.parser else text
                except ParseError:
                    if self._should_raise_on_parse_failure():
                        raise
                    parsed = PARSE_FAILED_PLACEHOLDER
                all_solutions[input_id].append(parsed)
                parallel_ids[input_id].append(par_id)
                sequential_ids[input_id].append(0)

        outputs: List[SequentialSamplingResult] = []
        for input_id in range(len(inputs)):
            outputs.append(
                SequentialSamplingResult(
                    solutions_per_prompt=all_solutions[input_id],
                    raw_generations=raw_generations[input_id],
                    parallel_ids=parallel_ids[input_id],
                    sequential_ids=sequential_ids[input_id],
                )
            )
        return outputs
