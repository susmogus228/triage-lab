# IT v2: external real-ticket sample and annotation limits

## Source and provenance

Mohammad Abdellatif, *Help Desk Tickets*, Mendeley Data version 3 (2026), DOI [10.17632/btm76zndnt.3](https://doi.org/10.17632/btm76zndnt.3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The publisher describes actual tickets and customer-support messages from one international software company in 2007–2023; some fields were masked by the source. Source masking is not a guarantee that the records contain no sensitive data. Text and source identifiers remain local in an ignored directory and must not be published without a separate privacy and license assessment.

The source used here is the [tasksource reporter-messages mirror](https://huggingface.co/datasets/tasksource/help-desk-tickets), pinned revision `d08adc88a5d39b938da00dc837ac613d9be02153`, reporter_messages/train.parquet. The mirror describes 357 tickets with reporter text, one row per issue; it excludes assignee replies, but multiple reporter messages may be combined. Its `issue_type` values are business workflow types and **not labels in this project's six-category taxonomy**. The downloaded content was checked against the pinned revision; SHA-256 and sampling metadata appear in `data/it_v2_external/sample_manifest.json` (local only).

## Sampling

357 rows have nonempty, exact-normalized-unique text. A fixed 160-row sample is constructed by sorting hashes of normalized text, shuffling with seed 726310, and assigning alternating records to development and test (80 each). Re-running the local preparation accepts identical outputs and fails rather than replacing changed evidence. Duplicate checks concern normalized exact text only: tickets may still be related by customer, organization, text template, dialogue or topic. All records are from one different company, not a random cross-organization sample. These splits are not independent of the conversations inherited by the AI agents.

## Why this is not a labelled benchmark

The source does not provide compatible category labels. A blind annotator inspected only heavily masked, vocabulary-limited context; most cases involve domain-specific business applications, so that reduced view is insufficient to reliably distinguish the original six IT classes from out-of-scope or ambiguous cases. Fixed-vocabulary checks found no explicit occurrences of Active Directory, Microsoft 365 product or specific network-file-service terms in the selected sample; hardware and software vocabulary occurrences were sparse. These are keyword checks, not class assignments, and must not be used as surrogate ground truth. Insufficient context is not evidence that a ticket is unknown.

**No AI labels or OOD ground truth are provided.** The frozen `sample_unlabelled_private.jsonl` can support explicitly labelled distribution-shift or error-review work but cannot establish category accuracy, unknown detection recall, false alarm rate, or model superiority. A defensible held-out benchmark requires a clear taxonomy and independent semantic annotations, followed by an annotation-quality audit; no human collection has been started. No model predictions were inspected by the annotator.

## Reproduction and verification

1. Fetch the pinned parquet into `data/it_v2_external/source.parquet`, verify its SHA-256 against the local manifest, and install a compatible `pyarrow` to a local dependency directory.
2. Run `data/it_v2_external/prepare_sample.py` with the Python environment and the pyarrow DLL directory added locally to `PATH`.
3. The source sample is validated as nonempty and normalized-text unique; sample membership and SHA must match the saved manifest. Do not export the private JSONL or source data.

The sample is an external exploratory set only. It is not a replacement for a dataset of genuinely new customers with independently established labels.
