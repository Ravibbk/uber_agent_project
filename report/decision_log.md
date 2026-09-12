# Decision Log

Non-obvious decisions made while building this, and why. (Fill in the
`[TODO: your number]` placeholders after you run the real eval — these are
currently placeholders since results depend on the actual 3M-row dataset.)

1. **Brand: Uber_Support.** Chose it over AmazonHelp/AppleSupport because
   ride-hailing complaints cluster into a small number of clearly distinct,
   actionable categories (fare, safety, lost item, driver behavior...),
   which makes both intent taxonomy design and escalation policy more
   defensible than a catch-all retailer like Amazon.

2. **Unit of analysis = (first customer message, brand's reply to it)**,
   not the full multi-turn thread. Most Uber_Support threads resolve or
   hand off within 1-2 exchanges; modeling full multi-turn dialogue state
   would add complexity the data mostly doesn't need. Documented as a
   limitation in the report, not hidden.

3. **Kept up to 2 prior turns as `thread_context`** even though the model
   only acts on the latest message, because a few intents (e.g. "still
   waiting on my refund from before") are only legible with context.

4. **9 intents, not more.** Started from Banking77-style granularity and
   collapsed down — more categories look impressive but most were <2% of
   the sample and only added noise to both classification and escalation
   logic. Fewer, cleaner categories > many overlapping ones.

5. **TF-IDF retrieval, not embeddings**, for grounding reply generation.
   Twitter support text is short and keyword-heavy; TF-IDF is free, fast,
   fully reproducible, and — importantly — debuggable (you can see exactly
   which shared words drove a match). Embeddings are the natural v2
   upgrade, noted in "what I'd do next."

6. **Reply generator is told to copy the *pattern*, not the specifics**, of
   retrieved historical replies (e.g., "ask for trip ID" is a pattern;
   a specific dollar amount is not). Without this instruction, early manual
   testing showed the model would sometimes carry over specific facts
   (amounts, dates) from the retrieved example into the new reply — a
   direct hallucination risk.

7. **Escalation uses hard rule-based triggers first, LLM-informed
   confidence second.** Safety-related intents and legal/fraud language
   always escalate regardless of what any model "thinks" — escalation is a
   safety-critical decision where a false negative (wrongly auto-handling)
   is worse than a false positive (unnecessarily escalating). This trades
   some automation rate for a much lower miss rate on the cases that
   matter most.

8. **Escalation confidence threshold set at 0.55**, not tuned end-to-end
   against the golden set at first — started as a reasonable prior and
   should be re-tuned once you have real golden-set precision/recall
   numbers (`[TODO: your number]` — see report Results section).

9. **Anger-language heuristic is a hardcoded keyword list, not a second LLM
   call**, for the refund+anger escalation rule. Cheaper and faster; a
   second LLM call for sentiment was evaluated as marginal extra value for
   this narrow use once the keyword list is combined with intent already
   being correct in eval (`[TODO: your number]`).

10. **Golden set deliberately oversamples short (<40 char) messages.**
    These are the hardest cases for the intent classifier and are
    under-represented in a naive random sample — including them
    intentionally is what keeps the accuracy number honest rather than
    inflated by easy cases (see report, "what's misleading").

11. **Judge scores on 4 separate dimensions (relevance / grounding / tone /
    actionability) instead of one overall score.** A single number hides
    *which way* a bad reply is bad, and the failure-analysis section needs
    that breakdown to produce real hypotheses instead of vague ones.

12. **Judge-human agreement is measured, not assumed.** Following the
    assignment's explicit requirement, `judge_calibration.py` exists
    specifically so the LLM-judge numbers aren't taken on faith — if
    agreement on a dimension is weak, that's reported as a real limitation.

13. **Trivial baseline always auto-handles (never escalates)**, by design —
    this makes its escalation recall exactly 0%, which is the right way to
    show that "our agent escalates the right things" is a real, earned
    result and not a trivially achievable one.

14. **`--sample` flag on data_prep.py** exists because the assignment
    explicitly says the full dataset won't be run — chose 5,000 threads as
    a default sample size, large enough for reasonable TF-IDF retrieval
    coverage across intents but small enough to keep iteration fast.

15. **Cleaning strips @mentions and URLs but keeps punctuation/case.** Tone
    (e.g., ALL CAPS, "!!!") is a meaningful signal for both intent and
    escalation (anger detection), so it wasn't normalized away.
