# Evolution experiment: review detection

## What changes

This experiment compares the existing topic classifier and review rule with an additional separately trained binary review detector. It does not modify the active application, retrain the topic classifier, or perform automatic actions on tickets. `g1` and `g2` are independent model generations, not a genetic algorithm or an automatic self-learning system.

## Sources and scope

- Bitext Customer Support LLM Chatbot Training Dataset, pinned revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443`, CDLA Sharing 1.0; hybrid synthetic. It is fetched separately following ML_DATA.md. The original source is not committed.
- CLINC150 `clinc/oos-eval` [revision 828f809](https://github.com/clinc/oos-eval/tree/828f809), CC BY 3.0 ([source](https://huggingface.co/datasets/clinc/clinc_oos)); crowdsourced task prompts, not real customers. Only training utterances with the explicit allowlist in `artifacts/evolution_data_manifest.json` are used as proxy out-of-scope. CLINC's own OOS labels are not transferred wholesale: source and project intent scopes differ. Their relevance to the project's categories is not independently adjudicated and ambiguous labels can remain.
- Both development and new final holdout are AI-authored synthetic examples, not customer traffic. The final holdout is balanced by design and cannot estimate operational prevalence or production precision. No demographic attribute data are available; demographic fairness cannot be established.

## Data and generation

The existing Bitext split yields 14,408 training sources. Source IDs are SHA-256 of normalized texts, partitioned BEFORE composites into 11,521 G1 fit and 2,887 feedback sources, stratified by broad category. CLINC source IDs are partitioned too. No source ID appears in both portions. The code rejects overlaps; this controls exact normalized duplicates, not near-duplicates or semantic template overlaps.

The fit set has 37,680 examples: 9,000 single statements, 9,000 same-broad-category composites, 18,000 distinct-category composites, and 1,680 filtered CLINC samples. Feedback has 8,420 examples using the same proportions and distinct source IDs. These are synthetic controlled examples; combination based on distinct *broad categories* does not reliably detect two intents from the same category. The exact number of unique generated examples is not necessarily equal to the number of rows; repeated source/template combinations remain possible.

Connectors including `and`, newline, `;` and others are applied to both same-category and cross-category composites to reduce trivial delimiter shortcuts. This does not prove shortcut removal or semantic label correctness. [BlendX](https://aclanthology.org/2024.lrec-main.218/) explains why synthetic separators can mislead multi-intent models. [Li et al.](https://aclanthology.org/2024.lrec-main.674/) studies hard negatives sharing in-scope vocabulary; the CLINC allowlist here is not a full set of such verified hard negatives.

## G1, G2, selection and freeze

G1 trains a word unigram/bigram TF-IDF + logistic binary classifier on fit only. G1 is then tested on the disjoint **training-feedback** partition, not on dev or the final test. G2 is trained from scratch with the same TF-IDF vocabulary on fit + feedback; fit samples and feedback examples classified correctly by G1 get weight 1, while the 2,324 feedback examples misclassified by G1 get weight 3. This is error-focused retraining with explicit labels, not gradient boosting, model merging, or live learning.

Previously viewed synthetic development and historical holdout cases (120) are **dev only**. The model and threshold are selected there using a predeclared gate: at least two fewer unflagged wrong/review-needed cases, coverage no lower than max(25%, baseline coverage minus 10 percentage points). Evaluation combines the existing baseline's 0.6 topic-score review rule OR the learned detector threshold. This threshold is a model score, not a calibrated probability.

An initial development report accidentally treated historical labels `MULTI` and `OUT_OF_SCOPE` as known topics. It is preserved as `artifacts/evolution_selection_invalid_v0.json`. Labels were normalized to `REVIEW`, results recalculated on dev, and the corrected `artifacts/evolution_selection.json` was frozen with SHA-256 for input sources, development, baseline and generations before the independent final test was read. No choices were made based on that test.

## Measured results

Development, 120 cases: baseline has 13 unflagged wrong/review-needed cases and 49/120 unflagged coverage (40.8%). G1 at score threshold 0.85 has 7 and 38/120 coverage (31.7%); it passes the predetermined dev gate. G2 at 0.85 has 8 and 37/120 coverage; it was not selected. This small development sample was used for selection; these results are not unbiased evidence of generalisation.

The separate 126-case holdout was independently authored by a different AI agent and read only after freezing selection. Known-topic correctness is 52/66 for both models because the topic model did not change. Baseline misses a review flag on 17/126 wrong or review-needed cases; G1 misses 10/126. Review flags on multi-topic cases rise from 16/30 to 22/30, while unknown cases stay at 25/26 and ambiguous at 4/4. Coverage falls from 38/126 (30.2%) to 24/126 (19.0%): a real tradeoff on this artificially balanced test, not evidence that the classifier's category accuracy improved. The 11.1-percentage-point reduction exceeds the size of the dev gate's 10-point tolerance, so promotion requires further independent validation and an operational decision. This comparison has not been tested on real customers.

A blind audit before predictions marked some holdout annotations potentially contestable, including EH048, EH032/034/035, EH070 and EH064; their original labels remain frozen. Any sensitivity analysis must be recorded separately and must not silently change the selection. On this synthetic holdout the input is a single text, whereas the application passes `subject + body`.

## Reproduce

For a fresh run, use a separate copy of the project and a new untouched holdout. The commands refuse to overwrite existing generated inputs, selection, models, or report. With the project Python environment and source CSV present, fetch the fixed CLINC data into `data/evolution_clinc_full.json` from `https://raw.githubusercontent.com/clinc/oos-eval/828f809/data/data_full.json`. Verify its SHA-256 from `artifacts/evolution_data_manifest.json` before reuse. Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m ml.evolution prepare
.\.venv\Scripts\python.exe -m ml.evolution select
# A fresh, independently authored and frozen holdout must be provided AFTER selection.
.\.venv\Scripts\python.exe -m ml.evolution final
.\.venv\Scripts\python.exe -m unittest ml_tests.test_evolution -v
```

Rerunning selection with knowledge of the existing final test turns that test into development. The current final result and checkpoint artifacts are historical evidence; do not present a tuned repeat as independently confirmed. Never load untrusted joblib files.

## Annotation sensitivity and task limits

A blind pre-evaluation audit recorded six contestable test labels (EH032, EH034, EH035, EH048, EH064, EH070). Their original labels and the frozen selection were retained. A separate sensitivity calculation stored in `artifacts/evolution_sensitivity.json` substitutes the alternate judgments without retraining or promotion. Under those alternatives, known-topic correct counts change from 52/66 to 49/62 for both versions; cases not flagged when wrong or review-needed are 17/126 for baseline and 11/126 for G1, rather than 17 and 10. These alternatives are not independently established truth.

Same-category composite examples are explicitly labeled as not requiring review during training, even though two distinct intents could occur within one broad category. This negative training signal limits the model and should be addressed before claiming general multi-intent support.

The final examples were authored by a separate AI agent without inspecting model outputs or training examples; independence from the full conversational context cannot be established. Treat results as diagnostic, not a blinded human study.
