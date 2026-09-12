# Report: Uber Support AI Agent

## 1. Problem framing

The system targets `@Uber_Support`, where a wrong answer can create safety,
financial, and reputational risk. Good means high recall for safety and legal
escalations, replies that do not invent trip facts, and replies that follow
historical resolution patterns rather than merely sounding plausible.

The nine intents are `fare_dispute`, `refund_request`, `driver_behavior`,
`safety_concern`, `lost_item`, `account_access`, `app_technical_issue`,
`cancellation_issue`, and `general_inquiry`. They are deliberately compact so
the labels are actionable and mutually understandable.

Not built: refund or account execution, authentication, persistent multi-turn
state, and embeddings retrieval. This prototype drafts a response and routes
it; it does not perform an irreversible customer action.

## 2. Evaluation protocol

Run `eval/build_golden_set.py --n 200` on a 5,000-thread TWCS sample. The
sampler stratifies by keyword proxy, samples across the date range, and
oversamples messages shorter than 40 characters. A human must label every row
with one intent and `AUTO_HANDLE` or `ESCALATE`, following the policy in
`src/escalation.py`. The final set must contain 150-250 hand-labeled rows.

`eval/run_eval.py` compares:

- **Trivial baseline:** most common gold intent, one canned reply, never escalates.
- **Simple baseline:** keyword intent rules, templates, safety-only escalation.
- **Agent:** offline/OpenAI classifier, TF-IDF grounding, reply generator, and
  safety-first escalation.

The evaluator reports accuracy and macro-F1 for intent, precision and recall
for escalation, and four 1-5 reply dimensions: relevance, grounding, tone, and
actionability. `eval/judge_calibration.py` compares the LLM judge with
human scores on 30-40 shared examples using Spearman correlation and mean
absolute difference.

### Current reproducibility status

The committed sample is a ten-pair smoke fixture. Its offline pipeline run is
intended to prove installation and wiring only; it is not a golden evaluation
and no production-quality headline number is claimed from it. The report must
be updated with the real 150-250 row hand-labeled results and calibration
numbers before submission. This is explicit to avoid presenting proxy labels or
an offline rubric as evidence.

## 3. Failure analysis checklist

After `run_eval.py`, inspect the five lowest-scoring or misclassified rows.
Record the real message, predicted intent/reply, expected label, and a
testable hypothesis. Expected risk areas are:

1. short messages with too little lexical signal;
2. messages combining fare and cancellation issues;
3. safety complaints without explicit safety vocabulary;
4. sarcasm and indirect driver complaints;
5. poor TF-IDF matches for rare product or policy terms.

These are hypotheses, not claimed findings; the final report should replace
them with examples from the completed golden set.

## 4. What is misleading about the headline number

The golden sample is intentionally stratified and oversamples hard short
messages, so its accuracy is not a traffic-weighted production estimate.
Escalation labels are human judgments rather than Uber policy ground truth.
The LLM judge can agree with a reply generator for the wrong reason, which is
why calibration is required. Finally, the ten-row bundled fixture is too small
to establish reliability, and offline fallback scores are not comparable to
API-backed judge scores.

## 5. What I would do with one more week

- complete and double-label the golden set, then adjudicate disagreements;
- compare TF-IDF with embeddings using a held-out retrieval benchmark;
- tune the confidence threshold against escalation cost, not accuracy alone;
- add adversarial safety and prompt-injection cases;
- add persistent multi-turn state and monitor automation rate, escalation
  misses, and reply edits after deployment.
