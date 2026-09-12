# Uber Support AI Agent

This project builds a simple customer-support agent for @Uber_Support. It:

- classifies each inbound customer message into a support intent
- retrieves similar past Uber support responses using TF-IDF
- drafts a short, support-style reply
- decides whether to auto-handle or escalate

The code is intentionally simple and readable so it can be understood and extended without a large ML or backend stack.

## Project overview

```text
uber-support-agent/
├── src/
│   ├── data_prep.py      # cleans raw Twitter support data into Uber threads
│   ├── intents.py        # intent taxonomy and LLM-based classification
│   ├── retrieval.py      # TF-IDF retrieval of similar historical replies
│   ├── reply_gen.py      # reply drafting based on retrieved examples
│   ├── escalation.py     # AUTO_HANDLE vs ESCALATE logic
│   ├── pipeline.py       # end-to-end agent runner
│   └── baselines.py      # simple reference baselines
├── eval/
│   ├── build_golden_set.py
│   ├── run_eval.py
│   ├── judge.py
│   ├── judge_calibration.py
├── data/
│   ├── sample/
│   │   ├── make_sample.py
│   │   └── twcs_sample.csv
│   └── raw/              # optional, not committed to Git
├── report/
│   ├── REPORT.md
│   └── decision_log.md
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE               # optional if you want to add one later
```

## Setup

```powershell
cd "C:\Users\LENOVO\Downloads\uber-support-agent"

py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then add your OpenAI key in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
CLASSIFY_MODEL=gpt-4o-mini
REPLY_MODEL=gpt-4o
JUDGE_MODEL=gpt-4o
```

For a quota-free local demo, run PowerShell with the offline fallbacks:

```powershell
$env:OFFLINE_MODE="1"
$env:JUDGE_OFFLINE="1"
```

This produces rule-based intent/reply suggestions and heuristic judge scores.
They are useful for smoke testing, but they are not evidence of LLM quality.

## Kaggle setup for the full dataset

If you want to use the real TWCS dataset, you need a Kaggle API token.

1. Create a Kaggle account at https://www.kaggle.com
2. Go to your account settings and click "Create New API Token"
3. Download `kaggle.json`
4. Place it in:
   - Windows: `C:\Users\<your-user>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
5. On macOS/Linux, lock down permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Then download the dataset:

```bash
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip
```

With the current Kaggle CLI, this creates `data/raw/twcs/twcs.csv`.

> If you do not want to use Kaggle, you can still run the sample pipeline using the bundled sample file in `data/sample/twcs_sample.csv`.

## Run with the sample data

This is the easiest way to verify the project works locally.

```bash
python src/data_prep.py --raw data/sample/twcs_sample.csv --out data/processed/uber_threads_sample.csv
python src/pipeline.py --threads data/processed/uber_threads_sample.csv --limit 10 --out data/processed/pipeline_sample_output.csv
```

## Run with full Kaggle data

After the Kaggle download step above, run:

```bash
python src/data_prep.py --raw data/raw/twcs/twcs.csv --out data/processed/uber_threads.csv --sample 5000
python src/pipeline.py --threads data/processed/uber_threads.csv --limit 20 --out data/processed/pipeline_output.csv
```

## Evaluation workflow

The evaluation is intentionally a two-step process. The script creates a
labeling file; you must hand-label 150-250 examples before reporting results.

```bash
python eval/build_golden_set.py --threads data/processed/uber_threads.csv --n 200 --out eval/golden_set_TO_LABEL.csv
```

Open the generated CSV in a spreadsheet and fill in `gold_intent`,
`gold_escalate`, and optional `labeling_notes` for every row. Save the result
as `eval/golden_set.csv`, then run:

To speed up review, you can first create transparent rule-based suggestions:

```bash
python eval/prefill_golden_set.py
```

This writes `eval/golden_set_PREFILLED.csv`. Review and correct those
suggestions in a spreadsheet, then save the reviewed file as
`eval/golden_set.csv`. The suggestions are not human labels and must not be
submitted without review.

```bash
cd eval
python run_eval.py --golden golden_set.csv --threads ../data/processed/uber_threads.csv
```

For judge calibration, hand-score 30-40 of the same rows using the rubric in
`eval/judge.py`, save them as `eval/human_judge_subset.csv`, and run:

```bash
python judge_calibration.py
```

Do not claim full assignment completion until `golden_set.csv`, the evaluation
output, and `human_judge_subset.csv` contain real human labels and scores.

## Notes

- This project is designed to be easy to understand and explain in interviews or GitHub demos.
- The retrieval layer uses TF-IDF instead of a heavier vector database to keep the solution lightweight and accessible.
- Generated datasets, virtual environments, and local secrets should not be pushed to GitHub.

## GitHub-ready checklist

Before pushing:

- keep `.env` out of Git
- keep `venv/` out of Git
- avoid committing large raw data files
- keep only source code, sample data, and documentation

This project is ready to push once the local environment and API key are set up correctly and the sample pipeline runs successfully.
