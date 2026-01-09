# src/seqsampling/sampling/sampler.py
from dataclasses import asdict, dataclass
from typing import Any, List

from ..models.base import LLMClient
from ..prompts.schema import PromptSchema, PromptContext
from ..parsing.base import Parser, ExtractionParser
from ..parsing.extractors import RemoveThinkingParser
from .config import SamplingConfig, GenerationConfig, ConstrainedGenConfig

@dataclass
class SequentialSamplingResult:
    # List over prompts; each element is list of K solutions
    solutions_per_prompt: List[Any]
    raw_generations: List[str]
    parallel_ids: List[int]    # per-solution parallel id within each prompt
    sequential_ids: List[int]  # per-solution sequential id within each parallel call

@dataclass
class SequentialSampler:
    client: LLMClient
    sampling_config: SamplingConfig
    constrained_config: ConstrainedGenConfig | None = None
    string_extractor: ExtractionParser = RemoveThinkingParser()  # default extractor
    parser: Parser = None  # to be set in __post_init__
    prompt_schema: PromptSchema = None  # to be set in __post_init__

    def run(self, inputs: List[str]) -> List[SequentialSamplingResult]:
        raise NotImplementedError("SequentialSampler is an abstract base class.")
    
    def _extract_answer(self, text: str) -> str:
        return self.string_extractor.extract(text)
    
    def _constrained_extra_params(self) -> dict:
        if self.constrained_config is None:
            return {}
        return self.constrained_config.to_params()

    def _should_raise_on_parse_failure(self) -> bool:
        """
        When constrained JSON mode is enabled, parsing failures should abort.
        Otherwise we can downgrade to warnings and placeholders.
        """
        if self.constrained_config is None:
            return False
        return bool(self.constrained_config.json_mode or self.constrained_config.json_schema)
