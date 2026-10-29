# Author-created synthetic holdout v1

## Purpose and provenance

`holdout_v1.jsonl` is a small, deliberately varied, manually enumerated **AI-author-created synthetic** set of English messages and expected outcomes. It is not a random sample of users, not sourced from actual customer tickets, and not labelled or verified by an independent human. A separate agent authored the examples without inspecting model predictions, the training examples or development examples. This reduces direct tuning to model outputs but does not establish independence from general knowledge or guarantee that similar text is absent from training. Compare normalized exact text and report any overlap before evaluation. Never use these rows to train, tune thresholds, choose a model or change a policy, and then call the same rows a fresh holdout.

There are 60 intentionally balanced, non-probability-sampled cases: four for each of 11 known categories and 16 marked `REVIEW`. The mix is constructed for diagnostic coverage, not to represent the real frequency of different kinds of requests. Each category has only four observations, so per-category proportions are unstable; the aggregate depends strongly on the chosen mix. Numeric uncertainty intervals under random-sampling assumptions do not establish population-level generalisation here. The independent test remains exploratory and provides no estimate of real-world reliability, fairness or operational performance.

## Annotation contract

Each JSONL row has an identifier, original `text`, expected outcome, scenario, and brief rationale. Expected outcomes use the supplied project taxonomy:

- `ACCOUNT`: account and profile access or details.
- `CANCEL`: an early contract termination charge, not cancellation of a purchase.
- `CONTACT`: asking to speak with a support person.
- `DELIVERY`: available delivery methods and options, not parcel tracking.
- `FEEDBACK`: a comment or suggestion about the experience.
- `INVOICE`: obtaining or correcting an invoice document.
- `ORDER`: an existing purchase, including cancelling a purchase.
- `PAYMENT`: payment failures, declines, or duplicate payment charges.
- `REFUND`: reimbursement of a purchase or its timing.
- `SHIPPING`: changing a shipping address for an existing order.
- `SUBSCRIPTION`: promotional newsletter opt-out.
- `REVIEW`: out-of-scope, insufficiently specific, or two distinct supported requests in one message. Multiple questions that are not independent intents need not be classified as multiple requests; each row has a reason.

Some real-world language overlaps these narrow labels; these annotations are a single-author operational judgment, not ground truth established by user studies. The taxonomy intentionally sends parcel tracking, contract cancellation without clarification, unrelated subjects and two distinct intents for review rather than forcing a class. A second independent human adjudicator and data from actual use would be required to establish label reliability and realistic performance.

## Pre-specified evaluation

1. Freeze the chosen model and review policy before reading predictions on this file; record artifact hashes, command, version and timestamp.
2. For the 44 single-label known cases, report classification accuracy and per-category counts. Separately count cases flagged for review, and accuracy among known cases routed without review; report the accepted denominator.
3. For the 16 `REVIEW` cases, report how many were actually sent for review. Do not count the model's forced topic as a successful `REVIEW`; `needs_review` is the decision. Break this down by `unknown`, `ambiguous`, and `multi-intent` counts.
4. Show the individual errors, decisions and denominators; distinguish an incorrect accepted classification from a correct classification flagged for review. Predictions and confidence are model scores, not calibrated probabilities.
5. If findings motivate further tuning, this set becomes development data. Future claims of held-out improvement require another independently created and frozen test set or, preferably, a properly consented and labelled real-data evaluation.
## Post-evaluation annotation caveat

The original annotation and protocol above are preserved. After the holdout was evaluated, a review found that source intent `track_order` belongs to `ORDER` and `delivery_period` belongs to `DELIVERY`, so case H059 (tracking a parcel) can reasonably be considered in scope. The original REVIEW annotation remains frozen. See `ml/STRESS_EVALUATION.md` and `artifacts/stress_sensitivity.json` for a separately identified post-hoc alternative, without changing model selection.
