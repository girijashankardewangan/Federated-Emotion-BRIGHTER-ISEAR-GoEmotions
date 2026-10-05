#!/usr/bin/env python3

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_csv(name):
    with open(ROOT / name, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

results = read_csv('results.csv')
go = read_csv('goemotions_results.csv')

assert len(results) == 60
assert len(go) == 20
assert len(results) + len(go) == 80

for rows in [results, go]:
    for row in rows:
        for metric in ['macro_f1', 'micro_f1', 'exact_match']:
            if metric in row and row[metric] != '':
                value = float(row[metric])
                assert math.isfinite(value)
                assert 0 <= value <= 1

print('PASS: 80 historical rows verified.')
print('PASS: metrics are finite and within [0,1].')
