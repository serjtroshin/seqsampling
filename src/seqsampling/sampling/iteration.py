from dataclasses import asdict, dataclass
import json
from typing import Any, List, Optional, Type
import warnings
from seqsampling.sampling.sampler import SequentialSampler, SequentialSamplingResult
from seqsampling.prompts.schema import PromptSchema, PromptContext, ChatMessage
from pydantic import BaseModel, ValidationError

from seqsampling.parsing.base import Parser, ParseError, PARSE_FAILED_PLACEHOLDER

@dataclass
class IterationPromptContext(PromptContext):
    input_text: str
    k: int
    previous_solutions: list[str] = None  # for iterative sampling

@dataclass
class JsonPromptSchema(PromptSchema):
    system_instruction: str
    field_name: str = "solution"  # configurable via config
    example_json: str = '{"solution": ["<Solution>. The answer is <Answer>."]}'  # example for clarity

    def build_messages(self, ctx: IterationPromptContext) -> List[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    "You must output a JSON object with a field "
                    f"'{self.field_name}', which is an array of exactly "
                    f"{1} item. Example: {self.example_json}\n\n"
                    f"Your solutions must be different from previous solutions: {str(ctx.previous_solutions)}\n\n"
                    f"Task input:\n{ctx.input_text}"
                ),
            ),
        ]

@dataclass
class XmlPromptSchema(PromptSchema):
    system_instruction: str
    tag_base: str = "solution"  # configurable tag prefix
    previous_solutions: list[str] = None  # for iterative sampling

    def build_messages(self, ctx: IterationPromptContext) -> List[ChatMessage]:
        tags = "\n".join(
            f"<{self.tag_base}>...</{self.tag_base}>"
        )
        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    f"Produce exactly 1 solution.\n"
                    f"Wrap it as:\n{tags}\n\n"
                    f"Your response should be different from previous solutions: {str(ctx.previous_solutions)}.\n\n"
                    f"Task input:\n{ctx.input_text}"
                ),
            ),
        ]

class JsonSolutionsParser(Parser):
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
class IterationSampler(SequentialSampler):
    """
    Sampler that generates multiple solutions for each input by making a single call to the model.
    """ 
    def __post_init__(self):
        if self.prompt_schema is None:
            self.prompt_schema: PromptSchema = JsonPromptSchema(
                system_instruction=self.system_instruction
            )
            self.parser = JsonSolutionsParser(key=self.prompt_schema.field_name)

    def run(self, inputs: List[str]) -> List[SequentialSamplingResult]:
        """
        inputs: list of input texts
        For each input, request K solutions in one model call and parse them.
        """
        print(f"Running IterationSampler on {len(inputs)} inputs with n={self.sampling_config.generation_config.n}, k={self.sampling_config.k}")

        raw_generations: List[List[str]] = [[] for _ in inputs]
        all_solutions: List[List[Any]] = [[] for _ in inputs]
        parallel_ids: List[List[int]] = [[] for _ in inputs]
        sequential_ids: List[List[int]] = [[] for _ in inputs]

        # Simple sequential version (could be parallelized with asyncio/threading)
        for turn_id in range(self.sampling_config.k):
            message_batches = [
                self.prompt_schema.build_messages(
                    IterationPromptContext(input_text=inp, k=self.sampling_config.k, previous_solutions=all_solutions[i])
                )
                for i, inp in enumerate(inputs)
            ]
            for input_id, messages in enumerate(message_batches):
                extra = self.sampling_config.extra_params or {}
                extra = {**extra, **self._constrained_extra_params()}
                gens = self.client.generate(
                    messages=messages,
                    **asdict(self.sampling_config.generation_config),  # n parallel calls per prompt
                    extra_params=extra,
                )
                for i, gen in enumerate(gens):
                    text = gen.text
                    raw_generations[input_id].append(text)
                    text = self._extract_answer(text)  # extract relevant part if needed
                    try:
                        new_solution = self.parser.parse(text)
                    except ParseError as e:
                        if self._should_raise_on_parse_failure():
                            raise
                        warnings.warn(f"Parsing failed for generation {i}: {e}")
                        new_solution = PARSE_FAILED_PLACEHOLDER
                    all_solutions[input_id].append(new_solution)
                    parallel_ids[input_id].append(i)
                    sequential_ids[input_id].append(turn_id)

        outputs: List[SequentialSamplingResult] = []
        for input_id in range(len(inputs)):
            outputs.append(SequentialSamplingResult(
                solutions_per_prompt=all_solutions[input_id],
                raw_generations=raw_generations[input_id],
                parallel_ids=parallel_ids[input_id],
                sequential_ids=sequential_ids[input_id],
            ))
        return outputs
