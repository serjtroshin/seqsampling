# examples/json_sequential_sampling.py
from seqsampling.models.openai_client import OpenAIClient
from seqsampling.sampling.config import SamplingConfig, GenerationConfig, ConstrainedGenConfig
from seqsampling.sampling.enumeration import EnumerationSampler, JsonPromptSchema, JsonSolutionsParser

def main():
    client = OpenAIClient(model="gpt-4.1")

    prompt_schema = JsonPromptSchema(
        system_instruction="You are a helpful assistant that generates diverse solutions.",
        item_model=None,
        example_json='{"solutions": ["Idea 1", "Idea 2", "Idea 3"]}',
    )
    parser = JsonSolutionsParser(key=prompt_schema.field_name, item_model=None)
    sampling_cfg = SamplingConfig(k=3, generation_config=GenerationConfig(n=1, max_tokens=512, temperature=0.9))
    # constrained generation to ensure JSON output with string items
    constrained_cfg = ConstrainedGenConfig(json_mode=True, json_schema=prompt_schema.response_json_schema())

    sampler = EnumerationSampler(
        client=client,
        sampling_config=sampling_cfg,
        constrained_config=constrained_cfg,
        prompt_schema=prompt_schema,
        parser=parser,
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
