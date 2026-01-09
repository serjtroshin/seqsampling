from dataclasses import asdict, dataclass
import json
from typing import Any, List, Optional, Type
import warnings

from seqsampling.sampling.sampler import SequentialSampler, SequentialSamplingResult
from seqsampling.prompts.schema import PromptSchema, PromptContext, ChatMessage
from pydantic import BaseModel, ValidationError

from seqsampling.parsing.base import Parser, ParseError, PARSE_FAILED_PLACEHOLDER

@dataclass
class JsonPromptSchema(PromptSchema):
    system_instruction: str = "You are a helpful assistant."
    field_name: str = "solutions"  # configurable via config
    example_json: str = '{"solutions": ["<Solution_1>. The answer is <Answer_1>.", "<Solution_2>. The answer is <Answer_2>.", ]}'  # example for clarity
    item_model: Optional[Type[BaseModel]] = None  # model for each item in the solutions list

    def build_messages(self, ctx: PromptContext) -> List[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    "You must output a JSON object with a field "
                    f"'{self.field_name}', which is an array of exactly "
                    f"{ctx.k} items. Example: {self.example_json}\n\n"
                    f"Task input:\n{ctx.input_text}"
                ),
            ),
        ]

    def response_json_schema(self) -> dict:
        """
        Return a JSON schema describing the expected response shape:
        {"solutions": [ <item> , ... ]}
        """
        item_schema = (
            self.item_model.model_json_schema()
            if self.item_model is not None
            else {"type": "string"}
        )
        return {
            "type": "object",
            "properties": {
                self.field_name: {
                    "type": "array",
                    "items": item_schema,
                }
            },
            "required": [self.field_name],
        }

@dataclass
class XmlPromptSchema(PromptSchema):
    system_instruction: str
    tag_base: str = "solution"  # configurable tag prefix

    def build_messages(self, ctx: PromptContext) -> List[ChatMessage]:
        tags = "\n".join(
            f"<{self.tag_base}_{i+1}>...</{self.tag_base}_{i+1}>"
            f"</{self.tag_base}_{i+1}>"
            for i in range(ctx.k)
        )
        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    f"Produce exactly {ctx.k} solutions.\n"
                    f"Wrap them as:\n{tags}\n\n"
                    f"Task input:\n{ctx.input_text}"
                ),
            ),
        ]

class JsonSolutionsParser(Parser):
    def __init__(
        self,
        key: str = "solutions",
        item_model: Optional[Type[BaseModel]] = None,
    ):
        self.key = key
        self.item_model = item_model

    def parse(self, text: str) -> List[Any]:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}") from e

        if self.key not in obj:
            raise ParseError(f"Missing key '{self.key}' in JSON.")

        items = obj[self.key]
        if not isinstance(items, list):
            raise ParseError(f"'{self.key}' should be a list.")

        if self.item_model is None:
            return items

        parsed_items = []
        for idx, item in enumerate(items):
            try:
                parsed_items.append(self.item_model.model_validate(item))
            except ValidationError as e:
                raise ParseError(f"Item {idx} failed validation: {e}") from e
        return parsed_items

@dataclass
class EnumerationSampler(SequentialSampler):
    """
    Sampler that generates multiple solutions for each input by making a single call to the model.
    """ 
    def __post_init__(self):
        if self.prompt_schema is None:
            self.prompt_schema: PromptSchema = JsonPromptSchema()
            self.parser = JsonSolutionsParser(key=self.prompt_schema.field_name)

    def run(self, inputs: List[str]) -> List[SequentialSamplingResult]:
        """
        inputs: list of input texts
        For each input, request K solutions in one model call and parse them.
        """
        print(f"Running EnumerationSampler on {len(inputs)} inputs with n={self.sampling_config.generation_config.n}, k={self.sampling_config.k}")

        # Build all message sets
        message_batches = [
            self.prompt_schema.build_messages(
                PromptContext(input_text=inp, k=self.sampling_config.k)
            )
            for inp in inputs
        ]

        extra = self.sampling_config.extra_params or {}
        extra = {**extra, **self._constrained_extra_params()}
        gen_cfg = self.sampling_config.generation_config

        # batched call: one OpenAI request per input, run concurrently under the hood
        batched_gens = self.client.generate_batch(
            batch_messages=message_batches,
            **asdict(gen_cfg),
            extra_params=extra,
        )

        outputs = []
        for gens in batched_gens:
            # `gens` is List[Generation] for a single input
            raw_generations: List[str] = []
            all_solutions: List[Any] = []
            parallel_ids: List[int] = []
            sequential_ids: List[int] = []

            for i, gen in enumerate(gens):
                text = gen.text
                raw_generations.append(text)
                text = self._extract_answer(text)  # extract relevant part if needed
                try:
                    solutions = self.parser.parse(text)
                except ParseError as e:
                    if self._should_raise_on_parse_failure():
                        raise
                    warnings.warn(f"Parsing failed for generation {i}: {e}")
                    solutions = [PARSE_FAILED_PLACEHOLDER] * self.sampling_config.k
                all_solutions.extend(solutions)
                parallel_ids.extend([i] * len(solutions))
                sequential_ids.extend(range(len(solutions)))

            outputs.append(
                SequentialSamplingResult(
                    solutions_per_prompt=all_solutions,
                    raw_generations=raw_generations,
                    parallel_ids=parallel_ids,
                    sequential_ids=sequential_ids,
                )
            )

        return outputs
