# Low-memory 192-token experiment

The local run used the previously pinned multilingual MiniLM weights and froze the encoder except its final layer plus the classification head. The maximum input length was 192, tokenizer padding dynamic for each input, microbatch size one, gradient accumulation four, and class-weighted loss normalization. The optimization was intended to reduce peak activation memory without changing the original dataset or removing class weighting. Gradient accumulation reduces per-step memory while preserving an effective larger update, though it can reduce throughput; see [Hugging Face documentation](https://huggingface.co/docs/transformers/grad_accumulation). `torch.no_grad` was used for development inference ([PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.no_grad)).

The script checks available physical RAM before start and throughout processing. A separate daemon watchdog stops only this Python process if available physical RAM falls below 700 MiB. The guard cannot reserve memory and cannot rule out brief system memory spikes. It does not close user programs or change OS paging settings. Existing reports/checkpoints are protected from overwrite.

## Actual outcome

The run started when a preceding observation reported about 3,339 MiB free. During initialization, the watchdog observed 654 MiB available and stopped the process. No epoch, checkpoint, or score was produced; neither the old model nor existing test data were changed. Raw watchdog evidence is `artifacts/it_lowmem_watchdog.json` and the derived status is `artifacts/it_lowmem_report.json`. The start observation and watchdog value are different moments and do not imply exact process-only peak memory consumption: other processes can change available RAM concurrently.

The earlier development set has already been used for model selection; its eventual result would be exploratory only. An unseen independent test remains necessary to support an improvement claim. This experiment does not implement or validate unknown-topic detection. Source data stay local. The demographic performance of the model cannot be assessed because protected demographic labels are absent.

## Running a new attempt

Use a separate output location/project copy; this script will not overwrite the evidence just produced. The guard allows start only when at least 1,600 MiB physical RAM is available and stops at 700 MiB; these are conservative operational values rather than a resource reservation. The pretrained model, Python dependencies, approved dataset and pinned model checksums must already be in place. Re-read `ml/IT_EXPERIMENT.md` and `ml/IT_LONG.md` for their source and restrictions.

```powershell
.\.venv\Scripts\python.exe -m unittest ml_tests.test_it_lowmem -q
.\.venv\Scripts\python.exe -m ml.it_lowmem
```
