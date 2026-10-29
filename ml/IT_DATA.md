# IT support tickets — data source and split

## Source and terms

Leonardo Santiago Benitez Pereira, *Classification of IT Support Tickets*, Zenodo record v2, DOI [10.5281/zenodo.7648117](https://zenodo.org/records/7648117). Zenodo's record API reports license **CC BY 4.0**. The publisher describes 2,229 real IT support requests from a Brazilian IT support provider, classified by three professionals in 2020 into seven categories. Author-described source languages are English, German, Portuguese and Spanish. The publisher describes automated, rule-based and manual removal of personal information; this has not been independently verified for every row.

Local raw files `data/it_support_raw/{X_train,X_test,y_train,y_test}.csv` are unchanged byte-for-byte after download and verified against the MD5 checksums published by the Zenodo record. `artifacts/it_data_manifest.json` records source hashes and output SHA-256 hashes. The prepared splits contain source text, should remain local and are not intended for screenshots or direct publication; even a dataset advertised as de-identified can contain residual sensitive data. A non-exhaustive regex audit is recorded as aggregate counts in `artifacts/it_data_privacy_pattern_counts.json`. A phone-like pattern matched 82 train and 77 test rows, and an IPv4-like pattern one test row; these patterns include false positives and do not establish whether an actual personal datum remains. No row values are exported in that audit.

## Frozen split and representation

`data/it_support_prepare.py` constructs:

- `data/it_support_prepared/train.jsonl`: 724 unique normalized texts for model fitting.
- `data/it_support_prepared/dev.jsonl`: 181 unique normalized texts for selecting model and threshold.
- `data/it_support_prepared/test.jsonl`: 657 original source test rows retained, including duplicate normalized texts.

Each JSONL row has `id`, `text`, `label`, `source_split`, `group_id`. Group IDs are connected components based on normalized exact strings or character 3–5-gram TF-IDF cosine similarity at least 0.95; they are **near duplicate groups**, not exact text hashes. For exact-normalized test groups, use SHA-256 of `" ".join(text.casefold().split())`. Fit downstream text vectorizers on train only.

The original source split has 621 duplicate normalized rows within train and 104 within test; 33 exact normalized groups occur in both train and test. One train exact-text group has conflicting category labels. To reduce leakage, unsupervised similarity is computed on all normalized text (including source test without labels), then train rows whose components touch source test and conflicting exact-label train groups are excluded. Source test is not used for model or hyperparameter selection. Remaining train is deduplicated by exact text and split with 5-fold `StratifiedGroupKFold`, shuffle and seed 42; first fold becomes dev. This predeclared transductive use of test **texts** for deduplication is not a fully untouched split: it is restricted to building similarity groups, not fitting features used by downstream models. Near groups are conservative; similarity below threshold and nonlexical semantic duplication remain possible. Counts and exact hash of each prepared file are in the manifest.

**Important: EOL is not a trainable known class in this leakage-controlled split.** Original train contains 45 EOL rows with only two distinct normalized texts; source test contains 58 EOL rows with the same two texts. Both groups overlap across source train/test and are removed from train. Thus train/dev have six known classes and original test has six known classes plus 58 EOL rows. Evaluating an ordinary supervised model as a seven-class closed-set classifier would be misleading. A fixed six-class benchmark may report EOL as an unseen-class subset, with a separate unknown/abstain analysis; sensitivity/alternative splits must preserve this limitation and cannot be advertised as independent evidence. Test is a preserved source split, not a time- or organization-held-out split.

The 657 original test rows correspond to 553 unique exact normalized texts and 535 near-duplicate components. Report both row-weighted metrics and metrics aggregated per exact group; do not call near-component weighting exact-text weighting. Demographic, timestamp, organization, worker and conversation identifiers are unavailable.

## Language estimates and limits

An optional local diagnostic `langdetect 1.0.9` seeded with 0 estimates source train English 1350, Portuguese 100, German 70, Spanish 12; source test English 590, Portuguese 34, German 22, Spanish 7, plus other labels. 96/2229 rows had top language probability < 0.8; 138 were under 30 characters. See `artifacts/it_data_language_estimate.json` for aggregates. This detector has not been validated on these technical snippets or redaction markers: counts are rough estimates, not language ground truth, and cannot establish translation or multilingual performance.

## Reproduction

From the project root, download the four exact filenames from the DOI to `data/it_support_raw/`, then run:

```powershell
.\.venv\Scripts\python.exe data\it_support_prepare.py
```

The script checks source checksums, preserves the source test, validates split group disjointness and only writes new outputs, or accepts a byte-identical rerun. Existing non-identical artifacts are never silently replaced. Language diagnostic was separately run with local `langdetect==1.0.9`; no labels from it enter the split or benchmark.
