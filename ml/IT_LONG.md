# Long-context IT classifier attempt

This bounded experiment was intended to compare a 192-token multilingual MiniLM partial fine-tuning run with the previously measured TF-IDF development macro F1 of 0.71618. The plan used the same frozen 724 training and 181 development examples, at most six epochs, the last encoder layer and classification head trainable, and best checkpoint selected only on development. The original IT test, the later synthetic blind diagnostic, and the unlabelled external sample would not be opened. This repeatedly used development set would remain non-independent; a truly fresh evaluation would be needed before an improvement claim. Unknown-request detection is a separate task and was not included.

## Observed result

On the attempted run, available physical memory was 1,162 MiB. The pre-start resource gate requires at least 1,600 MiB because the 471 MB pretrained model, PyTorch, activations and OS all require working memory; the previous 192-token attempt dropped the system below 500 MiB and caused substantial swapping. The gate therefore stopped before importing PyTorch or loading a model. **No epoch, checkpoint or model score was produced.** No user processes were closed. See `artifacts/it_long_report.json` for status and measurements. The 1,600 MiB gate is a conservative operational choice, not proof that all future starts above it will be safe.

The experiment monitors available physical memory before every batch and validation batch, stopping if it falls below 700 MiB; it also stops after a batch taking over 30 seconds. If executed, each completed epoch uses its own checkpoint instead of overwriting previous epochs. Existing report or checkpoints are protected against overwrite. For a new run use a fresh project copy with valid local source data and pinned pretrained assets. Always confirm actual available memory first. A post-start drop or a memory spike within one batch can still occur; limits are not a hard memory reservation.

A genuinely new independent test is needed before a promotion claim. The earlier 657-row source test was already inspected and its texts were used for deduplication without labels; the later AI-authored diagnostic has also been inspected. The task's private texts remain local. Protected saved IT models, selection and previously recorded metrics were unchanged.

## Source and tests

- [Pretrained multilingual MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) documents the base model and Apache 2.0 license.
- [Long document classification research](https://aclanthology.org/2021.nllp-1.22/) observes truncation can degrade performance in a different domain; it does not prove improvement for these IT tickets.
- `.venv\Scripts\python.exe -m unittest ml_tests.test_it_long_protocol -q` verified the memory gate, protected output paths and checkpoint comparison.
