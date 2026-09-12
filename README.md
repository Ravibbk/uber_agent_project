# Uber Support AI Agent

Take-home assignment implementation for Hiver SDE Intern. The agent is scoped to
`@Uber_Support` and performs three actions for each incoming message:

1. classify it into nine data-derived support intents;
2. retrieve similar historical Uber resolutions with TF-IDF and draft a reply;
3. return `AUTO_HANDLE` or `ESCALATE`, with an explicit reason.

The system is deliberately safe and reproducible: it works offline with
deterministic fallbacks, and optionally uses OpenAI for classification, reply
generation, and judging.

## Quickstart (under 15 minutes)

Requires Python 3.11+ and PowerShell on Windows (use `python` instead of
`py -3.13` on macOS/Linux).

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 src\data_prep.py --raw data\sample\twcs_sample.csv --out data\processed\uber_threads_sample.csv
py -3.13 src\pipeline.py --threads data\processed\uber_threads_sample.csv --limit 10 --out data\processed\pipeline_sample_output.csv
```

The last command writes ten agent decisions to
`data/processed/pipeline_sample_output.csv`. No API key is required for this
smoke run. The bundled sample contains 10 customer/brand pairs so the example
is fast, not statistically representative.

## Optional OpenAI mode

```powershell
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY
py -3.13 src\pipeline.py --threads data\processed\uber_threads_sample.csv --limit 10 --out data\processed\pipeline_openai_output.csv
```

Without a key, the classifier and reply generator use the documented offline
fallbacks. The judge also uses a deterministic smoke rubric; API-backed judge
scores must be used for the report.

## Full TWCS workflow

Download `thoughtvector/customer-support-on-twitter` from Kaggle into
`data/raw/twcs.csv` (raw data is intentionally gitignored), then run:

```powershell
py -3.13 src\data_prep.py --raw data\raw\twcs.csv --out data\processed\uber_threads.csv --sample 5000
py -3.13 eval\build_golden_set.py --threads data\processed\uber_threads.csv --n 200 --out eval\golden_set_TO_LABEL.csv
```

Read every sampled row and fill `gold_intent` and `gold_escalate` using the
taxonomy and escalation policy in `report/REPORT.md`; save the result as
`eval/golden_set.csv`. This hand-labeled 150-250 row file is the required
evaluation artifact and is intentionally not fabricated or committed.

```powershell
py -3.13 eval\run_eval.py --golden eval\golden_set.csv --threads data\processed\uber_threads.csv
```

For judge calibration, hand-score 30-40 of the same rows using the rubric in
`eval/judge.py`, save `eval/human_judge_subset.csv`, and run:

```powershell
py -3.13 eval\judge_calibration.py
```

## Repository map

- `src/data_prep.py`: cleans TWCS and reconstructs customer/reply pairs.
- `src/intents.py`: taxonomy, OpenAI classifier, and offline classifier.
- `src/retrieval.py`: transparent TF-IDF historical-resolution retriever.
- `src/reply_gen.py`: grounded reply generation and offline templates.
- `src/escalation.py`: safety-first routing policy.
- `eval/`: golden-set sampling, baselines, automated metrics, and judge calibration.
- `report/REPORT.md`: framing, evaluation protocol, limitations, and next steps.
- `report/decision_log.md`: non-obvious design decisions.

## Scope and limitations

This is a reply-drafting prototype, not an Uber account-action system. It does
not issue refunds, authenticate users, manage ongoing dialogue state, or claim
that the bundled ten-row smoke sample proves production quality. The report
requires real hand labels and human-vs-judge calibration before any headline
result should be submitted.

## Data and citations

Primary data: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter),
Kaggle user `thoughtvector`. Raw data is not redistributed here. The project
uses `scikit-learn` TF-IDF and OpenAI's API only when configured.

## GitHub checklist

- keep `.env`, `venv/`, raw data, and generated evaluation files out of Git;
- run the quickstart from a clean checkout;
- complete and commit the hand-labeled golden set only if its licensing and
  privacy review permits it;
- replace the report's smoke-test status with measured golden-set results.
