# Sequential Sampling for Diverse Responses
![CI](https://github.com/serjtroshin/sequential_sampling/actions/workflows/ci.yml/badge.svg?branch=main)
![Coverage](assets/coverage.svg)

This repository is a plug-and-play implementation of "Asking a Language Model for Diverse Responses" work presented at the Second Workshop on Uncertainty-Aware NLP at EMNLP 2025. For the paper original code, including reproduction of the original experiments on the gsm8k task, please visit https://github.com/serjtroshin/ask4diversity.

## Installation
To install the package, use poetry+conda:
```bash
conda create -n seqsampling python=3.10.15 -y
conda activate seqsampling
pip install poetry
poetry install
```

# Usage
You can run the examples provided in the `examples/` directory. For instance, to run the enumeration example:
```bash
python examples/enumeration.py
```
To see two-stage seeding + parallel expansion:
```bash
python examples/two_stage.py
```

The API is designed to be flexible and easy to use. You can create custom samplers, configure generation settings, and parse responses as needed.
```python
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
```
This script will return
```
Prompt 0:
  Solution 0: Users may churn because the app's content or features no longer meet their evolving needs or interests.
  Solution 1: Users might leave due to technical issues, such as frequent bugs, crashes, or compatibility problems with their devices.
  Solution 2: Users could churn because they perceive the subscription price as too high compared to the value they receive from the app.

Prompt 1:
  Solution 0: Breaking Language Barriers: Advances in Multilingual Natural Language Processing
  Solution 1: Building Intelligent Systems that Speak Any Language: The Rise of Multilingual NLP
  Solution 2: Unlocking Global Communication: How Multilingual NLP is Changing the World

Prompt 2:
  Solution 0: The argument lacks empirical support and relies heavily on anecdotal evidence, making its conclusions questionable.
  Solution 1: The reasoning in the argument commits a logical fallacy by assuming that correlation implies causation.
  Solution 2: The argument overlooks important counterexamples and alternative explanations, which undermines its overall persuasiveness.
```

## Two-stage seeding + parallel expansion
Generate a small batch of story premises with any sequential sampler (enumeration or iteration), then expand each premise in parallel with a templated prompt:
```python
from seqsampling.sampling import (
    EnumerationSampler, ParallelSampler, TwoStageSampler, SeedPromptTemplate,
    SamplingConfig, GenerationConfig, ConstrainedGenConfig,
)
from seqsampling.sampling.enumeration import JsonPromptSchema, JsonSolutionsParser
from seqsampling.sampling.parallel import JsonSolutionParser
from seqsampling.models.openai_client import OpenAIClient

client = OpenAIClient(model="gpt-4.1")

seeding_sampler = EnumerationSampler(
    client=client,
    sampling_config=SamplingConfig(k=3, generation_config=GenerationConfig(n=1, temperature=0.8)),
    constrained_config=ConstrainedGenConfig(json_mode=True),
    prompt_schema=JsonPromptSchema(system_instruction="Propose diverse story premises."),
    parser=JsonSolutionsParser(),
)

expansion_sampler = ParallelSampler(
    client=client,
    sampling_config=SamplingConfig(k=1, generation_config=GenerationConfig(n=2, temperature=0.9)),
    constrained_config=ConstrainedGenConfig(json_mode=True),
    parser=JsonSolutionParser(),
    prompt_schema=JsonPromptSchema(system_instruction="Expand the given seed into a vivid story beat."),
)

template = SeedPromptTemplate(
    template="Original: {input}\nSeed {seed_index}: {seed}\nWrite a short, vivid story paragraph inspired by this seed."
)

sampler = TwoStageSampler(
    seeding_sampler=seeding_sampler,
    expansion_sampler=expansion_sampler,
    prompt_template=template,
)

results = sampler.run([
    "Tell a magical realist story about a cafe where time moves backwards after midnight.",
    "Write a sci-fi tale about a city built on the back of a migrating creature.",
])
print(results[0].expanded_solutions)  # all expansions for every seed of the first prompt
```
The resulted output:
```
Running EnumerationSampler on 2 inputs with n=1, k=3
Running ParallelSampler on 6 inputs with n=2, k=1

Input 0: Tell a magical realist story about a cafe where time moves backwards after midnight.
Seeds:
  - A struggling poet discovers the Midnight Café, where after midnight, conversations with strangers unravel in reverse. Guests un-say their farewells, coffee refills itself, and regrets are swallowed unsaid. The poet learns to unmake a heartbreak, yet must choose whether to walk forward into dawn or undo his own arrival. The answer is: The poet decides to accept his past and leaves before dawn, cherishing the wisdom of both regret and renewal.
  - Every night at Lucida’s Café, the barista watches time retreat after midnight: spilled drinks leap into cups, tears are uncried, and confessions unspoken. One night, a mysterious patron arrives seeking to undo a tragic choice. As the hours slip backward, they must decide what is worth undoing, and whether the past should be rewritten. The answer is: They realize some memories, even painful ones, shape who they are and choose to let them remain.
  - In a quiet city corner, the café opens its doors to those who wish to relive cherished moments. After midnight, time rewinds only within its walls; lovers reunite, friendships rekindle, and lost opportunities are briefly restored. But, outside, the world moves on. A regular faces the dilemma: stay in the comfort of the reversed hours or step back into the unpredictable flow of time. The answer is: She bravely leaves, carrying the warmth of reclaimed memories into her future.
Expansions:
  Seed 0 (par=0): Under the honeyed glow of the Midnight Café, mugs filled themselves and laughter retreated into throats as words drifted backward through the thick, caffeinated air. The struggling poet, heart heavy with past wounds, watched as strangers unspoke confessions and pain unwound into silence. With each backward-spun hour, the ache of lost love loosened its hold, memories rewriting themselves softer—until only the choice remained: reclaim what was lost or step into morning’s uncertain light. As dawn blushed the sky, the poet quietly gathered his things and left, carrying forward not erased pain, but the bittersweet gift of understanding. The answer is: The poet decides to accept his past and leaves before dawn, cherishing the wisdom of both regret and renewal.
  Seed 0 (par=1): Under the flicker of dim bulbs, the Midnight Café hummed with an odd expectancy. At the stroke of twelve, words zipped back into mouths and laughter rewound into silence. The struggling poet sipped coffee that refilled itself, watching guests unwrite confessions and swallow unsettled regrets. He found his heartbreak softening, pain receding as if unspoken. Yet, as the sky paled with promise, he rose before dawn, choosing to carry his memories forward—a heart mended not by erasure, but by understanding. The answer is: The poet decides to accept his past and leaves before dawn, cherishing the wisdom of both regret and renewal.
  Seed 1 (par=0): At Lucida’s Café, midnight brings a secret reversal: steam curls into cooling mugs, broken hearts knit themselves whole as words inhale back behind lips, and laughter slides into silence, unsaid. Tonight, a stranger sits beneath the clock’s backwards sweep, eyes heavy with regret. As the night rewinds, their trembling hand closes around a letter they had once left behind. Time offers the impossible, but as old wounds fade and choices unmake themselves, the patron sees the loss in erasing scars that taught them tenderness. With dawn approaching, they clutch their memories close, choosing to let their pain remain—a bittersweet gift woven into their very being. The answer is: They realize some memories, even painful ones, shape who they are and choose to let them remain.
  Seed 1 (par=1): At Lucida’s Café, midnight cast a hush as clocks ticked backward, silver hands unspooling into yesterday. Espresso dribbles reversed, blending flawlessly into porcelain mugs as laughter rewound into quiet anticipation. On this particular night, the bell above the door chimed to herald a stranger, eyes shadowed with regret. The barista watched as the patron traced their night in reverse—words unsaid, sorrow undone—but as the city outside crawled back toward dusk, the stranger hesitated, clutching a photograph that no longer faded. In the gentle shimmer of time’s retreat, they realized the ache in their chest was memory’s gift, and with a trembling breath, chose to let the past remain, ordering coffee not as erasure, but as solace. The answer is: They realize some memories, even painful ones, shape who they are and choose to let them remain.
  Seed 2 (par=0): In the hush of the city’s sleeping hours, the café at the corner flickered to life, its golden lamplight spilling onto dew-slicked cobblestones. Past midnight, as clocks spiraled backward within, laughter and music blossomed anew—old friends found each other again, estranged lovers embraced, and conversations lost to regret unfurled as if they’d never been missed. Every backward-chiming hour gifted patrons the delicious ache of reclaiming what once slipped away. At her favorite window seat, Mira traced the steam on her cup and watched her reflection grow younger in the glass. The urge to linger was powerful, but as dawn pressed against the door, she rose. With a final glance at the café’s rewound magic, Mira stepped outside into the brisk, forward-marching morning, heart steadied by the delicate strength of memories relived. The answer is: She bravely leaves, carrying the warmth of reclaimed memories into her future.
  Seed 2 (par=1): As midnight’s chime faded, the café’s hands on the antique clock spun gently backward, and laughter from earlier in the evening bloomed anew. Porcelain cups refilled themselves, heartfelt confessions reversed into anticipatory glances, and a woman named Mara watched her once-lost friend step through the door again, radiant and unknowing. She whispered old jokes backwards, savoring the echo of love rediscovered. But outside the condensation-fogged windows, the city’s neon pulse beat forward. When Mara finally pressed through the velvet-curtained threshold, a rush of cool air met her face, carrying the scent of strong coffee and possibility. She felt the weight of reclaimed seconds in her chest—memories not to live in, but to live from. The answer is: She bravely leaves, carrying the warmth of reclaimed memories into her future.

Input 1: Write a sci-fi tale about a city built on the back of a migrating creature.
Seeds:
  - A young engineer discovers that the city atop the ancient migrating turtle-like creature is slowly poisoning their host with its waste. As the creature grows weak, the city’s continued existence is threatened. The answer is she rallies the citizens to invent sustainable living methods, forging a new symbiosis and ensuring both city and beast survive.
  - On the back of a colossal flying whale, an isolated city thrives, cut off from the rest of civilization for generations. When a daring explorer stumbles upon technology hinting at a world below, the city’s youth question everything they’ve been taught. The answer is a quest to reconnect with lost humanity, risking the stability of their aerial home.
  - A mysterious parasite outbreak sweeps through the city, affecting both inhabitants and the colossal beast they live upon. A renegade doctor theorizes the parasite is a message from the creature itself, trying to communicate with its passengers. The answer is deciphering the pathogen’s code, uniting science and empathy to heal both city and host.
Expansions:
  Seed 0 (par=0): Beneath endless stormswept skies, Tali knelt on the mossy expanse of Shell City, her hand pressed gently against the sluggish heartbeat thrumming through the ancient turtle-creature’s back. The city’s winding pipes dripped toxins into the beast’s cracks, and above, smokestacks curled black signatures of doom. Gathering the citizens atop the illuminated carapace, Tali revealed the peril: their magnificent moving world was dying beneath their feet. United by desperation and hope, inventors and dreamers worked through the night, crafting waste-purifying moss engines, water recyclers, and soaring gardens that fed both people and their gentle titanic companion. As the city’s lights glowed green for the first time, the turtle’s step grew lighter, carrying them all into a dawn of shared survival. The answer is she rallies the citizens to invent sustainable living methods, forging a new symbiosis and ensuring both city and beast survive.
  Seed 0 (par=1): Under the shadow of soaring crystalline towers, Mira knelt near the mottled shell, fingers tracing fissures oozing sick-green. For generations, her people called this moving giant home, but now its labored breaths rattled through the city foundations. With dread in her heart, Mira broadcast her findings to every citizen: their waste choked the ancient creature’s life. Rallying inventors and neighbors, she led a feverish campaign—converting refuse into fuel, weaving rooftops of moss and filtered water. As recycling engines hummed to life and aquaponic gardens flourished, the turtle’s strength returned. The city, once parasite, became partner; together, they migrated onward, symbiosis forging their shared future. The answer is she rallies the citizens to invent sustainable living methods, forging a new symbiosis and ensuring both city and beast survive.
  Seed 1 (par=0): Looming above endless clouds, the city of Zephyra clung to the ribbed, dusky hide of their ancient sky-whale, its spires and markets sheltered beneath the gentle sway of winged fins. For lifetimes, Zephyra’s people believed nothing existed beyond the mist, their legends cemented by the ceaseless flight. That myth shattered when Lira, a rebellious mechanic, unearthed a battered transmitter beneath the city’s generator—a voice crackling through the static, calling from the mysterious earth below. Hungry for answers, Zephyra’s youth gathered under neon lanterns, wrestling with forbidden curiosity, while elders whispered warnings: tampering could anger their colossal host. But the lure of rediscovered kinship outshone their fear, and as Lira led a secret expedition beyond the city’s edge, their fragile world trembled between hope and peril. The answer is a quest to reconnect with lost humanity, risking the stability of their aerial home.
  Seed 1 (par=1): At dawn, the city of Lume drifted on the whale’s misted spine above endless clouds, its crystal towers shimmering with dew. Generations had lived and died serenaded by the whale’s heartbeat, never daring to believe in the fabled world below—until Jessa unearthed the battered tablet, its screen flickering with maps of continents and oceans. Whispers swirled among the youth like stormwinds: If home was above, what waited beneath? Drawn by burning curiosity, a restless band prepared airships and vows in secret, the city’s elders warning that change would shatter their harmony. But against the backdrop of the whale’s mournful song, the quest began—a leap of faith toward the forgotten Earth, risking not just their exile, but the fragile balance that kept their skybound sanctuary aloft. The answer is a quest to reconnect with lost humanity, risking the stability of their aerial home.
  Seed 2 (par=0): Under the bioluminescent sky, panic surged through the city as crimson-veined spores drifted on the ever-shifting skin of the colossal beast beneath them. Homes shuddered with each of the creature’s uneasy heartbeats, its suffering mirrored in the fevered dreams of those infected. Dr. Lira Roe, cast out for her radical theories, watched patterns spiral in the parasite’s genome—utterances, she realized, of the behemoth’s pain and loneliness. In the heart of the city’s lab, she rallied a team, weaving empathy into code to craft a cure that was not merely medicine, but a message of understanding. When they released it, both city and host calmed, breathing as one—an alliance forged from the unlikeliest of conversations. The answer is deciphering the pathogen’s code, uniting science and empathy to heal both city and host.
  Seed 2 (par=1): Luminescent patterns erupted beneath the streets as fever swept the city, pulsing in sync with the shuddering breaths of the ancient beast supporting their world. Dr. Salyre dashed through the bioluminescent night, clutching slides of the writhing parasite: fractal tendrils in their microscopic structure mirrored language scripts lost to human memory. Racing against panic and quarantine, she assembled her ragtag team—engineers, bio-hackers, and empathetic children who could sense the beast’s moods—and together they tuned their sensors and their hearts. When the code finally unfolded—a plea for help, for companionship—the city’s neon lights blazed a response in harmony through the creature’s skin, ending the sickness in a wave of understanding. The answer is deciphering the pathogen’s code, uniting science and empathy to heal both city and host.
```

## Constrained generation tips
- `json_mode=True` is a light request for JSON output. For stricter enforcement, pass `json_schema` to build the OpenAI `response_format`.
- `item_model` (Pydantic `BaseModel`) describes each element in a JSON array. Set it when you want structured per-item validation; leave it `None` to accept plain strings.
- `json_schema` is the full response schema; use helpers like `JsonPromptSchema.response_json_schema()` (which includes `item_model` if provided) or handcraft your own for custom shapes.
- OpenAI strict schemas require `additionalProperties: false` on objects; the client adds this automatically when `json_schema` is strict.