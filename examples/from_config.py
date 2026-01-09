# examples/json_sequential_sampling.py

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from seqsampling.models.base import ChatMessage  # your existing type

from seqsampling.models.openai_client import OpenAIClient
from seqsampling.prompts.schema import PromptSchema
from seqsampling.parsing import Parser
from seqsampling.sampling.config import SamplingConfig, ConstrainedGenConfig
from seqsampling.sampling import EnumerationSampler

@hydra.main(config_path="../src/seqsampling/config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    client = instantiate(cfg.client)

    sampling_cfg: SamplingConfig = instantiate(cfg.sampling)
    constrained_cfg = ConstrainedGenConfig(json_mode=cfg.get("json_mode", False))

    sampler = EnumerationSampler(
        client=client,
        sampling_config=sampling_cfg,
        constrained_config=constrained_cfg,
        system_instruction=(
            "You are a helpful assistant that generates diverse solutions."
        )
    )

    inputs = [
        "Generate diverse hypotheses for why users churn from a subscription app.",
        "Generate diverse titles for a blog post about multilingual NLP.",
        "Generate diverse critiques of the following argument: ...",
    ]

    results = sampler.run(inputs)

    for i, sols in enumerate(results):
        print(f"\nPrompt {i}:")
        for j, s in enumerate(sols.solutions_per_prompt):
            print(f"  Solution {j}: {s}")

if __name__ == "__main__":
    main()