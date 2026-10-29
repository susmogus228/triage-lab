# IT v2: known-topic and unknown-request experiment

## Scope

This bounded experiment keeps the prior IT classifier and source data intact. It tries to improve known-category classification and to flag messages outside the six trained topics. Results are not demonstrated in production and do not justify replacing the conservative always-review IT workflow. No live customer messages are sent externally.

## Training, model selection and unknown detection

Five word+character TF-IDF logistic configurations were trained on the same 724 original training rows. Three-fold StratifiedGroupKFold was performed inside train; input features were fitted separately within each fold. The 181 original development cases were then used for model selection by macro F1. Their use in earlier model development makes this development result unsuitable as a new unbiased final metric. The chosen `balanced` variant has development accuracy 71.82% and macro F1 71.62%, the same as the prior IT model. The old 657-row test was not read or used for selection; it had already been inspected and its unlabelled text was used for near-duplicate grouping. The model choice, feature parameters, artifact SHA-256 and confidence policy are stored in `artifacts/it_v2_selection.json`.

The 192-token MiniLM training experiment was started and stopped during model initialization before optimization because available system memory fell below 0.5 GiB; no checkpoint or performance claim was produced. A longer-context score is therefore unavailable.

Two policies were compared on development: (1) one minus the maximum topic score, and (2) a binary TF-IDF classifier separating IT source train from 21 clearly unrelated CLINC150 intents. The training and validation lists come from their respective CLINC source splits, with exact normalized duplicate checks. CLINC labels relate to a different taxonomy, so these are proxy out-of-scope examples, not verified IT tickets. A threshold was selected on IT dev plus CLINC validation, requiring false flags on at most 40% of IT dev; among qualifying policies, highest proxy recall won. The chosen binary detector flagged 420/420 proxy CLINC validation utterances and 4/181 known IT dev examples. These easy source-distinguishing results do not demonstrate real-world unknown detection, calibrated probability, or stable false-flag rate.

## Synthetic blind diagnostic

After selecting model and threshold, a different AI agent authored a separate 100-case synthetic test (60 known, 30 UNKNOWN, 10 AMBIGUOUS). Its SHA-256 is in the aggregate report. The author did not inspect train/dev or model output, but the test is not a human-labeled customer sample; independence from wider conversation and semantic label correctness are not guaranteed. Exact text overlap with IT train/dev and CLINC train/val is checked, but paraphrases, semantic overlap and related expressions may remain.

The previous model and selected candidate use effectively the same known-topic TF-IDF classifier: both predicted 35/60 known topics correctly, macro F1 0.560. The current IT preview sends all cases for review (0% unflagged coverage, 30/30 unknown flagged). The experimental detector flagged 28/30 UNKNOWN and 8/10 AMBIGUOUS, but flagged 23/60 known and left 17/100 incorrect or review-needed messages without a flag; overall 41/100 did not receive a review flag. This small deliberately balanced set cannot estimate production prevalence, and its 38.3% known false-flag rate differs substantially from 2.2% on the easier proxy development mix. A flag is a research signal, not an automated external action.

An independent blind label review identified 18 contestable labels before predictions. The original frozen labels and selection remain unchanged. A post-hoc alternative treating these as AMBIGUOUS, not as proven ground truth, is recorded separately in `artifacts/it_v2_sensitivity.json`: known accuracy 30/53, unknown flagged 17/19, ambiguous flagged 21/28, known flagged 21/53 and missed errors/review cases 21/100. No raw case texts or IDs appear in the aggregate reports.

## Additional external real data without target labels

An additional Mendeley-derived sample, [`tasksource/help-desk-tickets`](https://huggingface.co/datasets/tasksource/help-desk-tickets), revision `d08adc88a5d39b938da00dc837ac613d9be02153`, contains actual help desk conversation messages. Its labels do not map to this classifier's categories and no trustworthy target annotations are available. From 80 held-out, unlabelled sampled texts the candidate flagged 48/80; this is **only a flag frequency**, not an accuracy, false-positive or unknown-recall measure. The original IT preview flags all 80. Masking has not been independently verified; source and sample text remain local and are excluded from reports. Source clients, organizations and languages cannot be assumed independent.

## Reproduction and safeguard

Run `.venv/Scripts/python.exe -m ml.it_v2 select` only in a fresh project copy with the local IT train/dev and CLINC data in place. The command refuses to overwrite existing model artifacts, selection or report. A fresh blind test authored without viewing model results is needed for each new iteration; after the test is inspected it is no longer independent for tuning. The current `final` command is pinned to the original test checksum and refuses to overwrite its saved report. `python -m ml.it_v2_sensitivity` separately reproduces the documented post-hoc label scenario in a fresh output location and also refuses overwrite. Use only locally generated trusted joblib artifacts; never load arbitrary model files.

References: [CLINC150 paper](https://aclanthology.org/D19-1131/), [OOD hard negatives](https://aclanthology.org/2024.lrec-main.674/), [OOD threshold non-transfer](https://aclanthology.org/2022.insights-1.3/), and [deduplication and ML leakage](https://scikit-learn.org/stable/common_pitfalls.html). Demographic fairness cannot be assessed because group labels are unavailable.

## Secondary diagnostic on the previously viewed IT source test

After freezing selection and the new synthetic diagnostic, the detector was applied to the already inspected original IT test. It flagged 3/599 known rows (3/551 unique exact texts) and 1/58 rows labelled EOL (1/2 unique exact texts). Only two distinct EOL texts mean this is not a representative sample of unknown IT requests, and the results are not a new independent test. The original test text was also used without labels for near-duplicate grouping. This result emphasizes that the model may accept unfamiliar IT requests without a review flag; the classifier forced a known class for all EOL examples. The previous IT workflow remains always-review. Aggregate counts are in `artifacts/it_v2_historical_diagnostic.json`; no source ticket texts are exported.

## Baseline provenance correction

A subsequent reviewer found that the v2 selection file did not itself freeze a SHA-256 of the historical baseline before this final comparison. The baseline's current SHA-256 is `6eb7ae8b2731b0b12d65dce807a83ca8dd10301e4d108b3fe7418f1e3c5f60bc`, matching the baseline hash in the earlier `it_selection_v2.json`; that corroborates identity but is not proof that v2 had independently sealed it before final evaluation. Original selection, report and sensitivity remain unchanged. Future runs of `select` save this hash and `final` rejects missing or changed baseline artifacts.
