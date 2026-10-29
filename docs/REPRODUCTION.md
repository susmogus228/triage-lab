# Reproducing TriageLab in an isolated workspace

Create a **new local copy of selected source files** under `work/reproduction/`. The generator does not copy the original data, email database, virtual environment, model weights, checkpoints, private tickets, or historical reports. The current `ml.training` refuses to overwrite an existing model or report; isolation also keeps new outputs and data separate from historical evidence.

Historical reports describe previous runs. This process writes new results in a separate directory; numbers may differ and are not a new independent evaluation. The same test set has already been inspected. Reusing it for tuning does not create an untouched test.

## Prerequisites

Windows x64, Python **3.11 x64**, Node.js and npm, plus sufficient RAM and space for a 206 MB CPU PyTorch wheel, 471 MB pretrained model, dependencies, data and checkpoints. The pinned wheel targets CPython 3.11/Windows x64. The commands assume `py -3.11` resolves to an installed interpreter; they do not install system packages. Python direct dependencies are pinned in `requirements.txt` and `ml/it_requirements.txt`; indirect packages are not locked. `npm ci` uses `frontend/package-lock.json`. Save `pip freeze` with each run to record the actual environment.

Run commands from the **original repository root** until told otherwise. If Python 3.11 exists but is not registered with the Windows launcher, substitute its absolute executable path for `py -3.11` in both commands. Network operations download public resources; training and ticket processing run locally. Respect the source licenses and attribution on redistribution: [Bitext / CDLA Sharing 1.0](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), [IT Support Tickets / CC BY 4.0](https://zenodo.org/records/7648117), and [multilingual MiniLM / Apache 2.0](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). Source IT tickets are described as de-identified by their authors, but anonymization is not independently verified. Keep raw and prepared tickets local.

## 1. Create a fresh copy

```powershell
py -3.11 -m ml.reproduce_workspace --name trial-01
Set-Location work\reproduction\trial-01
New-Item -ItemType Directory work, work\temp | Out-Null
$env:TEMP = (Resolve-Path work\temp).Path
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = (Join-Path (Get-Location).Path 'work\pip-cache')
$env:npm_config_cache = (Join-Path (Get-Location).Path 'work\npm-cache')
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use a unique name for each attempt. The generator refuses an existing target, including partial output. It copies an explicit source-file list and creates an empty `artifacts/`. From now on **run every command from the new workspace root**, including in new terminals. If a download or training fails, keep the partial output for inspection and create another workspace. Some upstream scripts overwrite their outputs on rerun. No command here deletes an existing workspace.

## 2. Restore the Bitext classifier

```powershell
$bitext = 'https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/430d1a89bd93bd1fa23c16f29dd53e73f0087443/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
Invoke-WebRequest $bitext -OutFile data\bitext.csv
if ((Get-FileHash data\bitext.csv -Algorithm SHA256).Hash -ne '6F81102B0100B97B8468EB04368033A23206BF1FDE9D53500D5806EC1001A434') { throw 'Bitext checksum mismatch; stop here.' }
.\.venv\Scripts\python.exe -m ml.training
```

The new `artifacts/ticket_classifier.joblib` and `artifacts/model_report.json` belong to this workspace. Training refuses to proceed if either output already exists; use another workspace for a new run. Results on synthetic English examples do not prove real customer performance. Only load trusted local joblib files.

## 3. Optional: reproduce the IT ticket experiment

Skip this section if you only want to run the main local demo. The IT classifier is available only after its separate data and model are prepared.

```powershell
New-Item -ItemType Directory data\it_support_raw | Out-Null
foreach ($file in @('X_train.csv','X_test.csv','y_train.csv','y_test.csv')) {
  Invoke-WebRequest "https://zenodo.org/records/7648117/files/$($file)?download=1" -OutFile "data\it_support_raw\$file"
}
.\.venv\Scripts\python.exe data\it_support_prepare.py
.\.venv\Scripts\python.exe -m ml.it_download_torch
.\.venv\Scripts\python.exe -m pip install 'work/it_wheels/torch-2.6.0+cpu-cp311-cp311-win_amd64.whl' -r ml/it_requirements.txt
.\.venv\Scripts\python.exe -m ml.it_download_pretrained
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 work\environment-freeze.txt
.\.venv\Scripts\python.exe -m ml.it_experiment baseline
.\.venv\Scripts\python.exe -m ml.it_experiment select
.\.venv\Scripts\python.exe -m ml.it_experiment refine
.\.venv\Scripts\python.exe -m ml.it_experiment seal
.\.venv\Scripts\python.exe -m ml.it_experiment final
```

Preparation checks the source CSVs against recorded MD5 values and records split checksums. The wheel and pretrained download scripts verify pinned digests. These are expensive CPU experiments; avoid concurrent training on a low-memory machine. The download scripts create `work/` inside the copy. Run each phase once and stop on error; the IT experiment protects existing evidence against overwrite. `refine` reproduces the documented later selection while retaining the original selection separately. The new report is `artifacts/it_experiment_report.json`, with selections, seal and model files from the same run. Do not combine it with historical artifacts.

The experiment compares six supported categories. EOL has no training examples and is reported separately; unknown-ticket detection is not validated. The test set's unlabeled texts were used for near-duplicate grouping, so evaluation is transductive. The dataset represents one organization's historical tickets. Its scores do not establish production performance or generalization to other users.

### IT v2 research candidate: selection only

After the IT `baseline` phase above, a fresh workspace can reproduce candidate selection using the public CLINC source. This trains the candidate and detector but does not activate either in the app:

```powershell
Invoke-WebRequest 'https://raw.githubusercontent.com/clinc/oos-eval/828f809/data/data_full.json' -OutFile data\evolution_clinc_full.json
if ((Get-FileHash data\evolution_clinc_full.json -Algorithm SHA256).Hash -ne '36923C3705A59E08FE9C3883D8BC2DD966EF93E22CB78AC41171782A698D56E0') { throw 'CLINC checksum mismatch; stop here.' }
.\.venv\Scripts\python.exe -m ml.it_v2 select
```

[CLINC150](https://github.com/clinc/oos-eval/tree/828f809) is CC BY 3.0; its selected intents are proxy unknowns, not validated labels for this project. The generator does not copy the original synthetic blind test or private external sample. Consequently this guide does **not** reproduce the v2 final report or the unlabelled external diagnostic; `ml.it_v2 final` requires a particular frozen blind file and verifies its hash. A new independent test requires a separately designed evaluation rather than claiming the old diagnostic is unseen. The app's original conservative IT mode remains in use.

## 4. Run the local demo

[Mailpit's official instructions](https://mailpit.axllent.org/docs/install/) provide Windows binaries. Download the Windows x64 ZIP from its [official release page](https://github.com/axllent/mailpit/releases), verify the publisher's checksum for that exact release, and extract `mailpit.exe` into the **new workspace's** `work/mailpit/`. Mailpit stores test email locally. Use synthetic messages, not real inboxes or personal data.

At the workspace root, build the frontend:

```powershell
Push-Location frontend
npm ci
npm run build
Pop-Location
```

Start Mailpit in one PowerShell window, at the workspace root:

```powershell
.\work\mailpit\mailpit.exe --smtp 127.0.0.1:1025 --listen 127.0.0.1:8025 --database work\mailpit\mailpit.db
```

In a second window, at the **same workspace root**:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8781
```

Open `http://127.0.0.1:8781/` and Mailpit at `http://127.0.0.1:8025/`. If a port is occupied, identify the existing process before starting another instance; the backend expects Mailpit on ports 1025 and 8025. The app uses its own local database and generated models; it is not publicly hosted.

## Verification boundary

Unit tests cover the copy list, names, empty artifacts, excluded data and refusal to overwrite. A source-only copy was additionally smoke-tested locally by copying the previously downloaded, checksum-verified source data, reusing the existing project Python environment, training the Bitext model and IT TF-IDF baseline, and creating a ticket through the API. This is not a clean dependency installation: fresh downloads, installation, full neural training, frontend build and the full demo still need separate verification on a new installation. Keep provenance and the actual generated report with any results, and review output for private data before sharing.
