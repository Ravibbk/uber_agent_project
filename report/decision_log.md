# Architecture Decision Records (ADR) & Decision Log

Non-obvious decisions, trade-offs, and design rationales made during the construction and upgrade of the TraceGuard platform.

---

### Core Architecture & Domain Decisions

1. **Brand Selection: `@Uber_Support`**  
   *Decision:* Chose Uber support over general retail (Amazon/Apple).  
   *Rationale:* Ride-hailing complaints naturally cluster into well-defined, safety-critical categories (fare disputes, lost items, driver behavior, vehicular safety, account security). This makes intent taxonomy design, deterministic guardrails, and escalation routing far more defensible and safety-critical.

2. **Unit of Analysis: `(customer_message, brand_reply)` Pairs**  
   *Decision:* Modeled the primary customer inquiry and brand resolution rather than full multi-turn conversational state.  
   *Rationale:* Twitter support interactions predominantly resolve or transfer to private DM within 1–2 turns. Keeping the primary turn as the atomic unit reduced unnecessary conversational state complexity while keeping the system auditable.

3. **Thread Context Preservation (Up to 2 Prior Turns)**  
   *Decision:* Kept prior dialogue turns as `thread_context`.  
   *Rationale:* Follow-up complaints (e.g., *"Still waiting on my refund from yesterday"*) are ambiguous in isolation but readily classifiable with immediate prior context.

4. **Nine-Intent Single-Label Taxonomy**  
   *Decision:* Standardized on 9 distinct intents (`fare_dispute`, `refund_request`, `driver_behavior`, `safety_concern`, `lost_item`, `account_access`, `app_technical_issue`, `cancellation_issue`, `general_inquiry`).  
   *Rationale:* Avoided over-granular taxonomies (like Banking77's 77 classes), which introduce high overlap and noisy decision boundaries for support agents.

5. **TF-IDF Grounding Retrieval vs. Dense Vector Embeddings**  
   *Decision:* Used TF-IDF vectorization with cosine similarity for historical resolution retrieval.  
   *Rationale:* Twitter support messages are concise and keyword-dense. TF-IDF provides fast, deterministic, reproducible retrieval without external vector database infrastructure or API latency.

6. **Pattern-Extraction Prompting (Anti-Hallucination Guard)**  
   *Decision:* Prompt instructions explicitly instruct the LLM to adopt resolution *patterns* (e.g., "request trip ID", "direct to app Lost Items tab") rather than copying specific dollar figures, trip IDs, or driver names from retrieved examples.  
   *Rationale:* Prevents factual hallucination and false financial promises.

7. **Multi-Tier Escalation Architecture**  
   *Decision:* Multi-tier decision gate: (1) Implicit physical safety keywords -> (2) Hard-escalate intent rules -> (3) Legal/fraud language -> (4) Anger-in-refund heuristic -> (5) Confidence threshold.  
   *Rationale:* Safety decisions must never rely solely on an LLM's probability score. False negatives (failing to escalate a dangerous driver) are catastrophic compared to false positives (unnecessary human review).

8. **Escalation Confidence Threshold: 0.55**  
   *Decision:* Set confidence gate threshold at 0.55.  
   *Rationale:* Established as a conservative prior: ambiguous intents with confidence below 55% are routed to human operators.

9. **Deterministic Anger Heuristic vs. LLM Sentiment Classifier**  
   *Decision:* Used regex and curated keyword matching for refund dispute sentiment detection.  
   *Rationale:* Faster, deterministic, and eliminates secondary LLM inference latency/cost.

10. **Stratified Hard Edge-Case Golden Benchmark**  
    *Decision:* Deliberately included short inquiries (<40 chars), sarcastic remarks, prompt injections, and ambiguous messages in `eval/golden_dataset.py`.  
    *Rationale:* Real customer support is noisy; a benchmark must not be biased towards clean, easy examples.

11. **Multi-Dimensional Reply Quality Evaluation**  
    *Decision:* Scored replies across 4 distinct dimensions (*Relevance*, *Grounding*, *Tone*, *Actionability*) plus binary *Hallucination*.  
    *Rationale:* A single aggregate quality score masks the specific failure mode of a reply.

12. **Empirical Judge Calibration Framework**  
    *Decision:* Created `eval/judge_calibration.py` to calculate Spearman's rank correlation ($\rho$), MAE, and hallucination agreement between human and automated judges.  
    *Rationale:* Automated judge scores cannot be trusted without empirical correlation against human ground truth.

13. **Trivial Baseline: Constant Prior with Zero Escalation**  
    *Decision:* Trivial baseline predicts constant prior `general_inquiry` and never escalates.  
    *Rationale:* Yields 0.0% escalation recall, proving that agent escalation accuracy is genuinely earned.

14. **Subsampled Preprocessing Pipeline (`--sample`)**  
    *Decision:* Maintained fast sample processing for rapid testing and CI.  
    *Rationale:* Enables full end-to-end execution in CI environments in under 10 seconds.

15. **Case & Punctuation Preservation in Data Cleaning**  
    *Decision:* Stripped Twitter handles and URLs but preserved uppercase text and punctuation.  
    *Rationale:* Punctuation and capitalization contain strong emotional signals critical for escalation.

---

### TraceGuard Platform Upgrades & Defect Resolutions

16. **Decoupled Implicit Safety Scanner from Intent Classifier (Defect E Resolution)**  
    *Decision:* Added an independent `IMPLICIT_SAFETY_KEYWORDS` scanner in `src/escalation.py` that executes before confidence evaluation.  
    *Rationale:* Previously, *"My driver was swerving across lanes"* was classified as `driver_behavior` (confidence 0.95) and auto-handled without escalation. Safety detection must be decoupled from classifier vocabulary.

17. **Lazy OpenAI Client Initialization (Defect A Resolution)**  
    *Decision:* Replaced module-level client instantiation with lazy loading inside functions.  
    *Rationale:* Importing modules must never crash when `OPENAI_API_KEY` is absent. Guarantees 100% offline testability and frictionless CI execution.

18. **Eliminated Baseline Label Leakage (Defect C Resolution)**  
    *Decision:* Replaced `Counter(train_df['gold_intent']).most_common(1)` in `src/baselines.py` with a fixed constant prior.  
    *Rationale:* A baseline must never inspect evaluation split labels to pick its prediction.

19. **Retrieval Index Case Isolation (Defect D Resolution)**  
    *Decision:* Added `exclude_ids: set` parameter to `HistoricalResolutionIndex.retrieve()`.  
    *Rationale:* During evaluation runs, test case IDs are excluded from the retrieval candidate set, preventing data contamination.

20. **Structured JSONL Tracing (`src/tracing.py`)**  
    *Decision:* Built a lightweight, zero-dependency span recorder logging stage durations, model IDs, prompt version hashes, and error categories.  
    *Rationale:* Provides deep observability into pipeline stages without requiring external APM or observability SaaS tools. Failures in tracing never break the primary agent.

21. **Deterministic Offline Heuristic Judge**  
    *Decision:* Implemented keyword-grounded offline heuristics in `eval/judge.py` when `OPENAI_API_KEY` is not present.  
    *Rationale:* Allows CI suites and offline reviewers to evaluate reply quality without paid API credentials while clearly distinguishing heuristic scores from LLM judge outputs.

22. **FastAPI Web Dashboard with Interactive Span Telemetry**  
    *Decision:* Added `src/api.py` with Pydantic request/response validation and an embedded modern dashboard.  
    *Rationale:* Provides recruiters and engineers with an immediate playground to test edge cases and inspect execution traces in real time.
