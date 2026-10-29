"""Train and assess a category classifier on the Bitext sample dataset.

Run: python -m ml.training
The dataset is downloaded separately as explained in ML_DATA.md.
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'bitext.csv'
ARTIFACT = ROOT / 'artifacts' / 'ticket_classifier.joblib'
REPORT = ROOT / 'artifacts' / 'model_report.json'
SEED = 42


def normalize(text):
    return re.sub(r'\s+', ' ', re.sub(r'\{\{.*?\}\}', ' placeholder ', text.lower())).strip()


def has_known_content(text, vocabulary):
    words = re.findall(r'(?u)\b\w\w+\b', text.lower())
    return any(word not in ENGLISH_STOP_WORDS and word in vocabulary for word in words)


def clean_rows(rows):
    by_text = defaultdict(list)
    for text, label in rows:
        if text.strip() and label.strip():
            by_text[normalize(text)].append((text.strip(), label.strip()))
    result = []
    conflicting = 0
    duplicates = 0
    for group in by_text.values():
        if len({label for _, label in group}) != 1:
            conflicting += 1
            continue
        duplicates += len(group) - 1
        result.append(group[0])
    return result, {'duplicate_rows': duplicates, 'conflicting_groups': conflicting}


def split_indices(labels, groups):
    y = np.array(labels)
    group_array = np.array(groups)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    # The first fold is the untouched test; second fold is validation.
    folds = [test for _, test in splitter.split(np.zeros(len(y)), y, group_array)]
    return np.concatenate(folds[2:]), folds[1], folds[0]


def choose_threshold(scores, correct):
    # Optimize coverage subject to >=90% validation accuracy for accepted items.
    # An entirely separate test evaluates the chosen fixed rule.
    for threshold in np.linspace(0.30, 0.99, 70):
        selected = scores >= threshold
        if selected.any() and selected.mean() >= 0.20 and correct[selected].mean() >= 0.90:
            return float(round(threshold, 2))
    return 1.01  # No validated rule: send every item for review.


def train():
    existing = [str(path) for path in (ARTIFACT, REPORT) if path.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite existing model or report; use an isolated reproduction directory: ' + ', '.join(existing))
    if not DATA.exists():
        raise FileNotFoundError(f'Missing {DATA}; see ml/ML_DATA.md for download instructions')
    with DATA.open(encoding='utf-8-sig', newline='') as f:
        raw = [(row['instruction'], row['category']) for row in csv.DictReader(f)]
    rows, cleanup = clean_rows(raw)
    texts = np.array([row[0] for row in rows])
    labels = np.array([row[1] for row in rows])
    groups = [normalize(x) for x in texts]
    train_i, val_i, test_i = split_indices(labels, groups)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)),
        ('classifier', LogisticRegression(max_iter=300, random_state=SEED)),
    ])
    model.fit(texts[train_i], labels[train_i])
    val_probs = model.predict_proba(texts[val_i])
    classes = model.classes_
    val_scores = val_probs.max(axis=1)
    val_known = np.array([has_known_content(text, model.named_steps['tfidf'].vocabulary_) for text in texts[val_i]])
    val_scores = np.where(val_known, val_scores, 0.0)
    threshold = choose_threshold(val_scores, classes[val_probs.argmax(axis=1)] == labels[val_i])
    probs = model.predict_proba(texts[test_i])
    pred = classes[probs.argmax(axis=1)]
    scores = probs.max(axis=1)
    known = np.array([has_known_content(text, model.named_steps['tfidf'].vocabulary_) for text in texts[test_i]])
    accepted = (scores >= threshold) & known
    precision = float((pred[accepted] == labels[test_i][accepted]).mean()) if accepted.any() else None
    report = {
        'model_version': 'bitext-tfidf-logreg-v1',
        'dataset': {'name': 'Bitext Customer Support LLM Chatbot Training Dataset',
                    'url': 'https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset',
                    'revision': '430d1a89bd93bd1fa23c16f29dd53e73f0087443',
                    'license': 'CDLA-Sharing-1.0', 'nature': 'hybrid synthetic',
                    'sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
                    'raw_rows': len(raw), 'unique_usable_rows': len(rows), **cleanup},
        'split': {'method': 'stratified group folds by normalized exact text, seed 42',
                  'train': len(train_i), 'validation': len(val_i), 'test': len(test_i),
                  'note': 'Similar templates may still cross splits; this is not independent real-world validation.'},
        'threshold': {'value': threshold, 'selection': 'first validation score threshold meeting >=90% accepted accuracy and >=20% coverage',
                      'validation_coverage': float((val_scores >= threshold).mean())},
        'test': {'accuracy': float(accuracy_score(labels[test_i], pred)),
                 'macro_f1': float(f1_score(labels[test_i], pred, average='macro')),
                 'coverage': float(accepted.mean()), 'accepted_accuracy': precision,
                 'accepted_rule': 'score at or above threshold AND at least one known non-stopword',
                 'class_order': classes.tolist(),
                 'confusion_matrix': confusion_matrix(labels[test_i], pred, labels=classes).tolist(),
                 'classification_report': classification_report(labels[test_i], pred, output_dict=True, zero_division=0)},
        'notes': ['English synthetic data; no evaluation on real customer messages.',
                  'Scores are uncalibrated model probabilities; not guarantees of correctness.',
                  'Unknown topics may receive high scores; send uncertain predictions for manual review.',
                  'No urgency classifier; reviewer corrections do not automatically retrain this model.'],
        'versions': {'sklearn': sklearn.__version__, 'numpy': np.__version__},
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    # Only locally generated trusted artifacts are loaded by inference.
    joblib.dump({'pipeline': model, 'threshold': threshold, 'model_version': report['model_version']}, ARTIFACT)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'split': report['split'], 'threshold': report['threshold'], 'test': {k: report['test'][k] for k in ('accuracy', 'macro_f1', 'coverage', 'accepted_accuracy')}}, indent=2))


if __name__ == '__main__':
    train()
