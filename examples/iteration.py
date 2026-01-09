# examples/json_sequential_sampling.py
from seqsampling.models.openai_client import OpenAIClient
from seqsampling.sampling.config import SamplingConfig, GenerationConfig, ConstrainedGenConfig
from seqsampling.sampling.iteration import IterationSampler, JsonPromptSchema, JsonSolutionsParser


def response_schema(field_name: str = "solution") -> dict:
    """
    JSON schema describing: {"solution": [ "<string>" ]}
    """
    return {
        "type": "object",
        "properties": {
            field_name: {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 1,
            }
        },
        "required": [field_name],
    }


def main():
    client = OpenAIClient(model="gpt-4.1")

    prompt_schema = JsonPromptSchema(
        system_instruction="You are a helpful assistant that generates diverse solutions.",
        field_name="solution",
        example_json='{"solution": ["Idea 1"]}',
    )
    parser = JsonSolutionsParser(key=prompt_schema.field_name, item_model=None)
    sampling_cfg = SamplingConfig(k=3, generation_config=GenerationConfig(n=1, max_tokens=512, temperature=0.9))
    constrained_cfg = ConstrainedGenConfig(
        json_mode=True,
        json_schema=response_schema(prompt_schema.field_name),
    )

    sampler = IterationSampler(
        client=client,
        prompt_schema=prompt_schema,
        parser=parser,
        sampling_config=sampling_cfg,
        constrained_config=constrained_cfg,
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
