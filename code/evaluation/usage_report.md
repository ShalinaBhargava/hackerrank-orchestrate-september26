# Token Usage and Cost Report

Final full-dataset run that produced `output.csv`.

## How the model is used

- The decision engine (forecast, plan generation, ranking, validation, explanations) is deterministic Python and makes **no model calls** per request.
- A one-time evidence-extraction pass (`code/evidence.py`) uses the LLM to structure the 215 messages (batched, 30 per call) and to read the linked images. Results are cached in `code/cache/evidence.json`; re-running `python code/main.py` reuses the cache and costs 0 tokens.
- Requests evaluated: 250.

## Per model

| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) | Price source |
|---|---|---:|---:|---:|---:|---:|---|
| gemini | gemini/gemini-3.6-flash | 24 | 52,527 | 91,897 | 144,424 | 0.3840 | LiteLLM price table (litellm) |

## Per stage

| Stage | Calls | Input tokens | Output tokens |
|---|---:|---:|---:|
| images | 16 | 24,945 | 13,105 |
| messages | 8 | 27,582 | 78,792 |

## Overall

- Model calls: 24
- Input tokens: 52,527
- Output tokens (includes reasoning tokens billed as output): 91,897
- Total tokens: 144,424
- Average tokens per request: 577.7
- Estimated total cost: USD 0.3840
- Estimated cost per request: USD 0.001536

Notes: token counts are the provider-reported `usage` values captured for every call, including calls for
sample-request evidence. Failed attempts (e.g. HTTP 503/429) returned no usage and are not billed.
