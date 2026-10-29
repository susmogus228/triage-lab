# ML data and reproduction

The model learns broad support categories from Bitext's hybrid synthetic customer support dataset, which is published under [CDLA Sharing 1.0](https://cdla.dev/sharing-1-0/). Attribution: Bitext, [Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443`. The source retains provider license and notices. No raw data is redistributed here.

For a new run, first create an isolated source-only workspace using [the reproduction guide](../docs/REPRODUCTION.md). Training refuses to overwrite an existing model or report. In the new workspace, download the pinned CSV to `data/bitext.csv` and run `python -m ml.training`:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/430d1a89bd93bd1fa23c16f29dd53e73f0087443/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv' -OutFile data/bitext.csv
python -m ml.training
```

The script produces a trusted local `artifacts/ticket_classifier.joblib` and `artifacts/model_report.json`. Only load your own generated model artifacts: joblib files can execute code on loading. Training does not require a remote model or send ticket contents to external APIs.

The classifier detects only categories in its training set, not urgency. It is not a production classifier. English synthetic examples may resemble each other even across the split. The reported holdout is not a real-world customer study and the uncalibrated scores do not guarantee correctness. Human corrections are retained by the application but do not automatically retrain the model. Training and test metrics vary by dependencies and data revision; use the included report and reproduce before reporting results.

The threshold is chosen on validation before test evaluation. Exact normalized duplicates are grouped; ambiguous conflicting labels are discarded. This does not rule out semantic or template overlap. The dataset lacks trustworthy demographic group labels; no demographic fairness claim is made.
