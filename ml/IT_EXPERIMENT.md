# IT support ticket classification experiment

## Aim and scope

Compare a word-and-character TF-IDF logistic regression with a pretrained multilingual MiniLM on the same six known IT support categories. Models were trained and evaluated entirely locally; no user tickets were sent to a remote model. The dataset is [Classification of IT Support Tickets](https://zenodo.org/records/7648117), CC BY 4.0; see [IT_DATA.md](IT_DATA.md). Source descriptions are real 2020 tickets from a single IT company, de-identified by the authors, not necessarily representative of later traffic. Do not publish raw ticket text; the provided de-identification is not an independently verified privacy guarantee.

## Data and evaluation

The data worker removed exact and high-similarity cross-split duplicate source training rows, and divided train/dev by near-duplicate connected components: 724 train and 181 dev unique normalized texts. Original held-out source test contains 657 rows but 553 unique normalized text groups. **Test text without labels was used transductively to group near duplicates and remove overlapping train items**, so the evaluation is not a fully untouched test. The data preparation records checksums in `artifacts/it_data_manifest.json`. Both models share the identical split. EOL was fully removed from train to avoid leakage: all its training texts also appeared in test, and the original test contains 58 EOL rows derived from only **two unique texts**. EOL is not one of the six trained labels. Six-category test metrics exclude EOL; seven-category metrics explicitly include it as unsupported. EOL is not a validation of unknown-intent detection.

The primary selection metric was development macro F1 across the six trained labels (ties favour the smaller TF-IDF model). There were two original neural epochs, then four additional dev-guided epochs starting from the best initial checkpoint with a restarted optimizer. Both attempts and histories were retained in `it_selection.json` and `it_selection_v2.json`. The final selected checkpoint and inputs were sealed, and the test file checksum matched the initial data manifest before the final model evaluation. Model choice was not changed after test evaluation. This small development set was reused for six epoch checkpoints; its score can be optimistic.

The multilingual MiniLM model uses [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, Apache-2.0. Pretrained third-party weights were used, then **only the last encoder layer and a new classifier head were updated** on the IT data; the rest was frozen. The model was not trained from scratch and this is not full-encoder fine-tuning. The TF-IDF baseline uses train-fitted word and character vectors and balanced logistic regression.

MiniLM receives at most 96 tokens per input. Longer inputs were truncated: 117/724 train, 31/181 dev, 229/657 test rows. This is a substantial limitation and a possible reason for poorer scores; comparisons do not establish a general ranking of model families. Performance by language or demographic group has not been established. Predicted scores are uncalibrated, and unknown-intent detection has not been independently validated.

## Results

On the six known categories, TF-IDF achieved test row accuracy 75.63% and macro F1 70.71%, compared with 73.62% and 68.02% for MiniLM. On 551 known unique normalized test texts, accuracy is 74.23% vs 72.78%. The full seven-label test row accuracy is 68.95% vs 67.12% because the held-out EOL class has no training examples. These numbers describe one limited dataset/split; they do not prove production performance, independent real-time generalisation, fairness, or safety. See `artifacts/it_experiment_report.json` for the exact row, exact-unique-group and near-group counts and per-class results, without raw ticket texts.

The selected model remains a separate IT research mode: `ml.it_inference.predict` returns `needs_review=true` for every request, since no unknown-category review threshold was validated. It does not replace the Bitext model or claim automatic routing.

## Method and dependencies

- [Hugging Face multilingual MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2): pretrained multilingual text representations and license.
- [scikit-learn data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html): separate preprocessing and model selection from outcome evaluation.
- [Hugging Face sequence classification guidance](https://huggingface.co/docs/transformers/tasks/sequence_classification): supervised downstream classification.

Local additions: PyTorch `2.6.0+cpu`, Transformers `4.57.3`, safetensors `0.8.0`; base project dependencies remain in `requirements.txt`. The PyTorch CPU wheel was obtained from the official index, checked against `24c9d3d13b9ea769dd7bd5c11cfa1fc463fd7391397156565484565ca685d908`, and installed locally. Pinned pretrained weight SHA-256 is `eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b`; configs and tokeniser are separately verified in `ml.it_download_pretrained` and at model load. All weights remain local in `work/it_pretrained`.

Existing selection, checkpoints, seal and final report are protected from overwrite. For a new independent experiment, use a separate project copy with a fresh data split; `python -m ml.it_experiment baseline`, `select`, `refine`, `seal`, and `final` write new artifacts only. After viewing this test, reusing it for tuning invalidates an independent comparison. The initial report remains historical evidence. Never load untrusted joblib or PyTorch checkpoints.
