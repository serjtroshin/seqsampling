# tests/test_openai_integration_smoke.py
import os
import json
from unittest import result
import pytest

from seqsampling.models.openai_client import OpenAIClient
from seqsampling.sampling.config import SamplingConfig, GenerationConfig, ConstrainedGenConfig
from seqsampling.sampling.enumeration import EnumerationSampler

@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="OPENAI_API_KEY not set; skipping live OpenAI smoke test",
)
def test_openai_sequential_sampler_smoke():
    client = OpenAIClient(model="gpt-4.1")

    sampling_cfg = SamplingConfig(k=3, generation_config=GenerationConfig(n=2, max_tokens=512, temperature=0.9))
    constrained_cfg = ConstrainedGenConfig(json_mode=True)

    sampler = EnumerationSampler(
        client=client,
        sampling_config=sampling_cfg,
        constrained_config=constrained_cfg,
    )

    inputs = ["Give me 3 reasons why users churn from a subscription app.", "List 3 innovative uses of AI."]
    results = sampler.run(inputs)

    assert len(results[0].solutions_per_prompt) == sampling_cfg.generation_config.n * sampling_cfg.k
