# Stress evaluation and model comparison

This experiment separates **model selection** from an author-independent (not human-independent) final test. All examples are synthetic and the authors are AI agents; none represent real customer messages. Exact-deduplication cannot establish semantic independence or real-world validity. The test is a convenience sample with only four examples per known category.

## Steps and reproduction

1. `data/stress_dev.json` was authored for development by the experiment agent. It has 44 known single-intent cases, eight unknown cases, and eight multi-intent cases. `data/stress_test.json` was authored by the same agent and is supplemental illustrative material, **not** the final holdout and not used to select the model or calculate the reported final test.
2. Both models use the same Bitext training partition. Baseline is the original word n-gram TF-IDF logistic regression; candidate is a character n-gram TF-IDF logistic regression. No holdout examples are used for fitting.
3. The 0.6 raw-score review rule was already in the application. It is held fixed for both models, alongside the known-content heuristic. A single-word vocabulary match is not robust out-of-scope detection, and the scores are not calibrated probabilities.
4. `python -m ml.stress_eval select` evaluates development only. A candidate is selected only when known-intent accuracy increases by at least three percentage points **and** the number of wrong or inappropriate accepted decisions does not increase. The baseline, candidate, training source and dev SHA-256 values are saved in `artifacts/stress_selection.json` before final data are read. Running select again after viewing the holdout invalidates the holdout for a new unbiased comparison.
5. A different AI agent authored `data/holdout_v1.jsonl` without inspecting training/dev examples or predictions. Annotation rules and limitations are in `data/holdout_v1.md`. This data are synthetic and not independently labelled by people. `python -m ml.stress_eval final` checks the frozen hashes and exact normalized overlap against the source and development rows, then records both model results and individual decisions in `artifacts/stress_report.json`. No hyperparameter or selection decision uses final outcomes.
6. `artifacts/ticket_classifier.joblib` and `artifacts/model_report.json` remain the original active baseline; the candidate is `artifacts/stress_char_candidate.joblib`. Do not load a joblib artifact from an untrusted source.

## Results from this run

The baseline was selected on development: known-intent accuracy 35/44 (79.55%), eight unsafe automatically accepted cases. Character candidate was 34/44 (77.27%), nine unsafe automatic cases. Therefore the original deployment remains active.

On the separate 60-case holdout, the baseline classified 28/44 known examples correctly (63.64%); it routed 12/16 REVIEW cases for review. Of 17 known examples automatically accepted, 16 were correctly classified. There were five unsafe automatic cases in all: one known-intent error and four REVIEW examples incorrectly accepted. Breakdown of REVIEW: 8/9 unknown, 3/3 ambiguous, 1/4 multi-intent were sent to review.

The candidate scored 33/44 (75%) known classifications, routed 14/16 REVIEW cases to review and had three unsafe automatic cases. This is an observation on a small test, not a reason to change the frozen dev selection retroactively. Trying to select or modify a candidate from these results would consume this holdout; a new independent test is needed for a later improvement claim. Absolute differences are unstable because the sample is deliberately balanced and small.

`accuracy` and `macro_f1` refer only to the 44 known single-intent cases, regardless of review; `coverage` is the fraction of all 60 sent onward without review. `accepted_in_scope_accuracy` uses only known accepted cases, with denominator recorded in `counts`. `should_review_recall` is the proportion of REVIEW examples routed to review. `unsafe_auto_count` includes both incorrect accepted known cases and any accepted REVIEW case. Percentages should always be presented with denominators and alongside the synthetic-data limitation.

Bitext's labels carry particular meanings, not universal support taxonomy: `CANCEL` concerns termination fees; cancellation of a purchase is `ORDER`; `DELIVERY` is delivery options while `SHIPPING` concerns address changes; `SUBSCRIPTION` is newsletter enrollment. This mapping can make ordinary phrasing ambiguous and is not a universal standard.

## Methodological sources

- Larson et al., 2019: [An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://aclanthology.org/D19-1131/) describes why out-of-scope requests require separate evaluation.
- Li et al., 2024: [Generating Hard-Negative Out-of-Scope Data](https://aclanthology.org/2024.lrec-main.674/) evaluates difficult unknown requests with vocabulary shared with known intents.
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) explains why predicted scores need calibration before probability claims.

No demographic labels are present, so demographic fairness has not been measured. This work makes no production suitability or demographic performance claim.

## Post-hoc annotation sensitivity: H059

An independent reviewer identified an ambiguity: H059 asks why a specific parcel's tracking has not moved. The frozen holdout marks it `REVIEW` under the holdout's simplified taxonomy, while Bitext includes `ORDER/track_order` and `DELIVERY/delivery_period` examples. The original labels and primary results remain unchanged. As an explicitly post-hoc sensitivity analysis we reassigned **only H059** to `ORDER`, without changing the actual holdout file, model selection, probabilities, or review decisions. Whether `ORDER`, `DELIVERY` or manual review is the correct operational label requires an agreed taxonomy and independent human annotation.

The baseline's known-topic accuracy changes from 28/44 (63.64%) to 28/45 (62.22%); review recall changes from 12/16 (75%) to 11/15 (73.33%). The candidate's accuracy changes from 33/44 (75%) to 33/45 (73.33%); review recall changes from 14/16 (87.5%) to 13/15 (86.67%). Both models sent H059 to review with `REFUND` as the forced predicted topic, so the number of unsafe accepted cases is unchanged. The analysis is recorded in `artifacts/stress_sensitivity.json` and reproducible with `python -m ml.stress_sensitivity`. It is not a new independent holdout or justification to change the frozen model choice.

## Active inference versus development selection

`models[].selected_on_dev` records the frozen development selection. `models[].role` records the actual served artifact: baseline is `active` and candidate is `comparison`, even if a future experiment selects the candidate. The top-level `cases` always describes the active baseline. Selection itself does not update inference; model promotion requires an explicit, separately evaluated change.

This holdout supplies one `text` field per message. The live API concatenates a separately entered subject and body before classification, so these results do not separately validate that combined-input path.
