# examples/two_stage.py
"""
Two-stage sampling:
  1) Enumerate a handful of seeds.
  2) Expand each seed in parallel with a templated prompt.
"""
from seqsampling.models.openai_client import OpenAIClient
from seqsampling.sampling import (
    EnumerationSampler,
    ParallelSampler,
    TwoStageSampler,
    SeedPromptTemplate,
    SamplingConfig,
    GenerationConfig,
    ConstrainedGenConfig,
)
from seqsampling.sampling.enumeration import (
    JsonPromptSchema as EnumPromptSchema,
    JsonSolutionsParser,
)
from seqsampling.sampling.parallel import (
    JsonPromptSchema as ParallelPromptSchema,
    JsonSolutionParser,
)


def main():
    client = OpenAIClient(model="gpt-4.1")

    seeding_sampler = EnumerationSampler(
        client=client,
        prompt_schema=EnumPromptSchema(
            system_instruction="Propose diverse story premises."
        ),
        parser=JsonSolutionsParser(),
        sampling_config=SamplingConfig(
            k=3,  # three seeds per input
            generation_config=GenerationConfig(n=1, max_tokens=512, temperature=0.8),
        ),
        constrained_config=ConstrainedGenConfig(json_mode=True),
    )

    expansion_sampler = ParallelSampler(
        client=client,
        prompt_schema=ParallelPromptSchema(
            system_instruction="Expand the given seed into a vivid story beat."
        ),
        parser=JsonSolutionParser(),
        sampling_config=SamplingConfig(
            k=1,  # ParallelSampler enforces k=1; control fan-out via n
            generation_config=GenerationConfig(n=2, max_tokens=512, temperature=0.9),
        ),
        constrained_config=ConstrainedGenConfig(json_mode=True),
    )

    template = SeedPromptTemplate(
        template=(
            "Original request: {input}\n"
            "Seed {seed_index}: {seed}\n"
            "Write a short, vivid story paragraph inspired by this seed."
        )
    )

    two_stage_sampler = TwoStageSampler(
        seeding_sampler=seeding_sampler,
        expansion_sampler=expansion_sampler,
        prompt_template=template,
    )

    inputs = [
        "Tell a magical realist story about a cafe where time moves backwards after midnight.",
        "Write a sci-fi tale about a city built on the back of a migrating creature.",
    ]

    results = two_stage_sampler.run(inputs)

    for i, res in enumerate(results):
        print(f"\nInput {i}: {res.input_text}")
        print("  Seeds:")
        for seed in res.seeding_result.solutions_per_prompt:
            print(f"    - {seed}")

        print("  Expansions:")
        for exp in res.expansions:
            for j, sol in enumerate(exp.parallel_result.solutions_per_prompt):
                par_id = exp.parallel_result.parallel_ids[j]
                print(f"    Seed {exp.seed_index} (par={par_id}): {sol}")


if __name__ == "__main__":
    main()
