# Report: Uber Support AI Agent

## Evaluation status

The pipeline was evaluated on 198 Uber_Support customer/reply pairs sampled
from the TWCS data. The project owner reviewed and confirmed the intent and
escalation labels in `eval/golden_set.csv`. The reply judge ran in offline
fallback mode because no OpenAI judge key was available; those reply scores
are smoke-test evidence, not a substitute for human judge calibration.

## 1. Problem framing

The system targets `@Uber_Support`, where a wrong answer can create safety,
financial, and reputational risk. Good means high recall for safety and legal
escalations, replies that do not invent trip facts, and replies that follow
historical resolution patterns rather than merely sounding plausible.

The nine intents are `fare_dispute`, `refund_request`, `driver_behavior`,
`safety_concern`, `lost_item`, `account_access`, `app_technical_issue`,
`cancellation_issue`, and `general_inquiry`.

Not built: refund or account execution, authentication, persistent multi-turn
state, or embeddings retrieval. This prototype drafts a response and routes it;
it does not perform an irreversible customer action.

## 2. Results vs. baselines

The trivial baseline predicts the most common intent, uses one canned reply,
and never escalates. The simple baseline uses keyword intent rules, templates,
and safety-only escalation. The agent uses the classifier, TF-IDF grounding,
reply generation, and safety-first escalation.

| Metric | Trivial | Simple | Agent |
|---|---:|---:|---:|
| Intent accuracy | 16.7% | 60.1% | 100.0% |
| Intent macro-F1 | 0.032 | 0.610 | 1.000 |
| Escalation precision | 0.0% | 68.2% | 100.0% |
| Escalation recall | 0.0% | 75.0% | 100.0% |
| Reply relevance (1-5) | n/a | n/a | 3.81 |
| Reply grounding (1-5) | n/a | n/a | 3.00 |
| Reply tone (1-5) | n/a | n/a | 4.00 |
| Reply actionability (1-5) | n/a | n/a | 5.00 |
| Hallucination rate | n/a | n/a | 0.0% |

These values are reproduced by:

```powershell
py -3.13 eval\run_eval.py --golden eval\golden_set.csv `
  --threads data\processed\uber_threads.csv `
  --judge_out eval\judge_scores.csv
```

### Judge calibration

The evaluation produced 198 offline heuristic judge scores. A human-scored
30-40 row subset is still required to calculate Spearman agreement and
hallucination agreement. No human agreement number is claimed here because
creating one without independent human scores would be misleading.

## 3. Failure analysis

The confirmed labels produced no intent or escalation errors for this run, so
there are no observed misclassified examples to present as failures. The
following are the five highest-risk test categories identified for follow-up:

1. **Short messages:** “help” or “charged again” may lack enough signal.
   Hypothesis: lexical rules and TF-IDF need more context.
2. **Mixed intents:** a cancellation complaint may also request a refund.
   Hypothesis: the single-label taxonomy forces a priority choice.
3. **Implicit safety complaints:** “driver was swerving” may omit the word
   “unsafe.” Hypothesis: safety recall depends on expanding trigger coverage.
4. **Sarcasm:** “great, another late driver” is difficult for keyword rules.
   Hypothesis: sentiment and intent are expressed indirectly.
5. **Rare product terms:** unusual airport, promotion, or payment vocabulary
   may retrieve weak historical matches. Hypothesis: TF-IDF misses semantic
   similarity that embeddings could capture.

These are risk hypotheses, not fabricated error counts.

## 4. What is misleading about the headline number

The 100% agent intent score is not a production accuracy estimate. The sample
is stratified rather than traffic-weighted, and the confirmed labels were
created with the same offline intent logic that powers the no-key agent path.
This creates label leakage and likely inflates the score. Escalation labels are
also policy judgments, not Uber ground truth. Finally, offline judge scores
are not equivalent to an API-backed LLM judge, and there is no human-agreement
evidence yet.

## 5. What I would do with one more week

- independently double-label the golden set and adjudicate disagreements;
- hand-score 30-40 replies and run `judge_calibration.py`;
- compare TF-IDF with embeddings using a held-out retrieval benchmark;
- tune the escalation threshold against the cost of missed safety cases;
- add adversarial safety, sarcasm, prompt-injection, and multi-turn tests.
