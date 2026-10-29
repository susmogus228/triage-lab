"""Create a clean local reproduction workspace without copying private data.

Only explicit source files are copied; no virtualenv, databases, trained models,
original data, work directory, reports, secrets, or user submissions are copied.
"""
import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'requirements.txt',
    'backend/app.py',
    'data/it_support_prepare.py',
    'frontend/index.html', 'frontend/package.json', 'frontend/package-lock.json', 'frontend/vite.config.js',
    'frontend/src/format.js', 'frontend/src/it-report.js', 'frontend/src/it-v2-report.js',
    'frontend/src/main.jsx', 'frontend/src/style.css',
    'ml/__init__.py', 'ml/training.py', 'ml/inference.py', 'ml/evolution.py',
    'ml/it_experiment.py', 'ml/it_inference.py', 'ml/it_v2.py', 'ml/it_requirements.txt',
    'ml/it_download_pretrained.py', 'ml/it_download_torch.py',
)
RESERVED = re.compile(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$', re.IGNORECASE)

def validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name) or RESERVED.fullmatch(name):
        raise ValueError('Name must contain 1-64 letters, digits, underscores or hyphens and cannot be a Windows device name.')
    return name

def create_workspace(source, target):
    source = Path(source).resolve(strict=True)
    target = Path(target).absolute()
    allowed = (source/'work').resolve()
    if not allowed.is_relative_to(source) or not target.parent.resolve().is_relative_to(allowed):
        raise ValueError('Reproduction workspace must be under the project work directory.')
    sources = []
    for relative in FILES:
        path = source/relative
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source) or not resolved.is_file() or path.is_symlink():
            raise ValueError('Invalid source file: '+relative)
        sources.append((relative, resolved))
    # mkdir with exist_ok=False prevents replacing any previous workspace.
    target.mkdir(parents=False, exist_ok=False)
    for relative, path in sources:
        out = target/relative
        out.parent.mkdir(parents=True, exist_ok=True)
        # Destination is a newly created directory; source is not modified.
        with out.open('xb') as dst, path.open('rb') as src:
            shutil.copyfileobj(src,dst)
    (target/'artifacts').mkdir()
    (target/'data').mkdir(exist_ok=True)
    return target

def main():
    parser=argparse.ArgumentParser(description='Create a fresh isolated source-only reproduction workspace.')
    parser.add_argument('--name',required=True,help='Unique workspace name under work/reproduction')
    args=parser.parse_args()
    name=validate_name(args.name)
    parent=ROOT/'work'/'reproduction'
    parent.mkdir(parents=True,exist_ok=True)
    target=create_workspace(ROOT,parent/name)
    print('Created source-only workspace:',target)
    print('No data, trained weights, database, reports, virtualenv or credentials were copied.')
    print('See docs/REPRODUCTION.md in the original project for next steps.')

if __name__=='__main__':
    main()
