#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
repair = ROOT / 'repairs' / 'train_corrected.py'
patch = ROOT / 'repairs' / 'train_contracts.patch'
outdir = ROOT / 'data' / 'repository_analysis'
outdir.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

metadata = {
    'historical_train_py_preserved': (ROOT / 'train.py').exists(),
    'candidate_repair': str(repair.relative_to(ROOT)),
    'candidate_repair_sha256': sha256(repair),
    'patch': str(patch.relative_to(ROOT)),
    'patch_sha256': sha256(patch),
    'historical_results_preserved': {
        'results.csv': (ROOT / 'results.csv').exists(),
        'goemotions_results.csv': (ROOT / 'goemotions_results.csv').exists(),
    },
}

output = outdir / 'repair_build_metadata.json'
output.write_text(
    json.dumps(metadata, indent=2),
    encoding='utf-8'
)

print(f'Wrote {output}')