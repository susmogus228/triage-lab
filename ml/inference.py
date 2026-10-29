"""Inference against a trusted locally trained artifact, never uploaded pickle."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
from ml.training import has_known_content

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / 'artifacts' / 'ticket_classifier.joblib'
REPORT = ROOT / 'artifacts' / 'model_report.json'


@lru_cache(maxsize=1)
def _load():
    if not ARTIFACT.is_file():
        raise RuntimeError('Model is not trained yet. Run python -m ml.training.')
    return joblib.load(ARTIFACT)  # Only the locally generated artifact, not user input.


def predict(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Ticket text is required.')
    artifact = _load()
    model = artifact['pipeline']
    probabilities = model.predict_proba([text])[0]
    order = probabilities.argsort()[::-1]
    labels = model.classes_
    top = int(order[0])
    score = float(probabilities[top])
    known_features = has_known_content(text, model.named_steps['tfidf'].vocabulary_)
    return {'label': str(labels[top]), 'confidence': score,
            'alternatives': [{'label': str(labels[int(i)]), 'confidence': float(probabilities[int(i)])} for i in order[1:4]],
            'needs_review': (not known_features or score < artifact['threshold']),
            'model_version': artifact['model_version']}


def get_report() -> dict:
    if not REPORT.is_file():
        raise RuntimeError('Model report is not available. Run python -m ml.training.')
    return json.loads(REPORT.read_text(encoding='utf-8'))
