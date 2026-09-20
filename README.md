# eval-jevbench

`eval-jevbench` packages the 231 public JevBench v1.2 tasks as an Inspect/AnyEval eval. JevBench measures typed decision models: the model receives a state plus a bounded rubric and must return a probability distribution over the exact label set.

The included public tiers are:

- `original`: 72 tasks
- `easy`: 48 tasks
- `hard`: 111 tasks

Held-out tiers are not included: standard, judge, and the private halves of easy and hard are absent, so scores from this package cover only the 231 public tasks shipped here.

## Running

Install with the Inspect extra:

```bash
pip install -e '.[inspect]'
```

Run all public tasks:

```bash
OPENAI_BASE_URL=https://api.trustedrouter.com/v1 \
OPENAI_API_KEY=... \
inspect eval jevbench/jevbench --model openai/trustedrouter/trev-1.0
```

Convenience tasks are also exported as `jevbench_easy`, `jevbench_hard`, and `jevbench_original`.

## Gateway Translation

The solver does not call Inspect `generate()`. It reads the Inspect model name, strips an Inspect provider prefix where needed, and posts one typed decision request to `{OPENAI_BASE_URL}/decide`:

```json
{
  "state": "<task state>",
  "model": "<gateway model id>",
  "questions": {
    "decision": "<translated question>"
  }
}
```

Question translation follows the TrustedRouter decision API:

- `noul` becomes `{"type":"boolean","instructions":...,"criteria":{"true":...,"false":...}}`; `answers.decision.probability` is converted to `{"yes": p, "no": 1-p}`.
- `choice` becomes `{"type":"choice","instructions":...,"criteria":{label: description}}`, preserving upstream label order and falling back to the label text as its description.
- `score` becomes `{"type":"score","instructions":...,"criteria":[level_0_text, level_1_text, ...]}`; probabilities are keyed by level index strings.

The same code path handles native decision models and chat models. The scorer records `probs_source` as `native` when the served model's catalog modality is `text->decision`, otherwise `verbalized`.

## Scoring

Each task is valid only if the returned distribution covers exactly the label set, has numeric probabilities in `[0, 1]`, and sums to 1 within 2%. Distributions inside the 2% band are renormalized. Invalid or failed calls are scored incorrect with a fixed `invalid_reason`.

Per-task score metadata includes:

- `probs`
- `probs_source`
- `usage`
- `latency_s`
- `served_model`
- `brier`
- `top_confidence`
- `correct`
- `tvd_to_gold` when a gold distribution is present
- `ordinal_mae` for score questions

Inspect metrics include accuracy and mean Brier.

## Attribution

The task data comes from JevBench v1.2 by Fabian Standhartinger, MIT licensed, with model and benchmark context at <https://benchmarkheaven.com/jev-models>. Upstream reference files in this repository are used only for development context and are not shipped by the package.
