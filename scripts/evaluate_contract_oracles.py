import os
REPO_ROOT = "/content/Federated-Emotion-BRIGHTER-ISEAR-GoEmotions"
import numpy as np
import pandas as pd
import copy
import json

def sc1_original(model_values, val_seq):
    best_val = -1
    best_state = None
    for val, nv in zip(val_seq, [1, 2, 3]):
        for k in model_values:
            model_values[k] = nv
        if val > best_val:
            best_val = val
            best_state = model_values
    return best_state

def sc1_repaired(model_values, val_seq):
    best_val = -1
    best_state = None
    for val, nv in zip(val_seq, [1, 2, 3]):
        for k in model_values:
            model_values[k] = nv
        if val > best_val:
            best_val = val
            best_state = copy.deepcopy(model_values)
    return best_state

def run_sc1():
    fixtures = [
        ("early_max", [0.9, 0.4, 0.2], 1),
        ("later_max", [0.2, 0.8, 0.3], 2),
        ("tied_max", [0.8, 0.8, 0.3], 1),
    ]
    out = []
    for name, vals, exp in fixtures:
        m = {"w": 0}
        r = sc1_original(m, vals)
        op = all(v == exp for v in r.values())
        m = {"w": 0}
        r = sc1_repaired(m, vals)
        rp = all(v == exp for v in r.values())
        out.append({"fixture": name, "expected_selected": exp,
                    "original_pass": op, "repair_pass": rp})
    return out

def run_sc2():
    fixtures = [
        ("early_best", [0.8, 0.4, 0.2], 0.8),
        ("later_best", [0.2, 0.9, 0.3], 0.9),
    ]
    out = []
    for name, vals, exp in fixtures:
        orig = vals[-1]
        rep = max(vals)
        out.append({"fixture": name,
                    "expected_selected": exp,
                    "original_pass": abs(orig - exp) < 1e-9,
                    "repair_pass": abs(rep - exp) < 1e-9})
    return out

def original_isear_loader(df):
    """Retains codes 1-4 but always constructs 5 outputs (surprise inactive)."""
    df = df[df["EMOT"].isin([1, 2, 3, 4])].copy()
    for label in ["joy", "anger", "fear", "sadness", "surprise"]:
        df[label] = 0
    return df

def repaired_isear_loader(df):
    """Retains codes 1-4, only 4 outputs."""
    df = df[df["EMOT"].isin([1, 2, 3, 4])].copy()
    for label in ["joy", "anger", "fear", "sadness"]:
        df[label] = 0
    return df

def original_brighter_loader(df):
    """Silently fills missing columns with zero."""
    df = df.copy()
    for label in ["joy", "anger", "fear", "sadness", "surprise"]:
        if label not in df.columns:
            df[label] = 0
    return df

def repaired_brighter_loader(df):
    """Rejects missing columns and nonbinary values."""
    required = ["joy", "anger", "fear", "sadness", "surprise"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in required:
        vals = set(df[c].unique())
        if not vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Nonbinary values in {c}: {vals}")
    return df.copy()

def run_sc3():
    out = []

    # Fixture 1: ISEAR all 7 codes → 4 retained, but original adds surprise
    df = pd.DataFrame({"SIT": ["a"]*7, "EMOT": [1,2,3,4,5,6,7]})
    try:
        o = original_isear_loader(df.copy())
        # Original "passes" = retains 4 rows AND has surprise inactive
        # We define pass = correctly rejects unsupported label
        # Original adds 'surprise' → FAIL
        orig_pass = "surprise" not in o.columns
    except Exception:
        orig_pass = False
    try:
        r = repaired_isear_loader(df.copy())
        rep_pass = "surprise" not in r.columns and len(r) == 4
    except Exception:
        rep_pass = False
    out.append({"fixture": "isear_all_codes",
                "expectation": "no surprise column",
                "original_pass": orig_pass,
                "repair_pass": rep_pass})

    # Fixture 2: BRIGHTER missing surprise column → original fills, repair rejects
    df = pd.DataFrame({"joy": [1], "anger": [0], "fear": [0], "sadness": [0]})
    try:
        o = original_brighter_loader(df.copy())
        # Original silently fills → should FAIL
        orig_pass = False
    except Exception:
        orig_pass = True  # Good if it rejects
    try:
        r = repaired_brighter_loader(df.copy())
        # Should reject
        rep_pass = False  # If no exception, it's a failure
    except ValueError:
        rep_pass = True
    out.append({"fixture": "brighter_missing_col",
                "expectation": "reject missing column",
                "original_pass": orig_pass,
                "repair_pass": rep_pass})

    # Fixture 3: BRIGHTER nonbinary target → original accepts, repair rejects
    df = pd.DataFrame({"joy": [0.5], "anger": [0], "fear": [0],
                       "sadness": [0], "surprise": [0]})
    try:
        o = original_brighter_loader(df.copy())
        orig_pass = False  # Should fail (accepts 0.5)
    except Exception:
        orig_pass = True
    try:
        r = repaired_brighter_loader(df.copy())
        rep_pass = False
    except ValueError:
        rep_pass = True
    out.append({"fixture": "brighter_nonbinary",
                "expectation": "reject nonbinary target",
                "original_pass": orig_pass,
                "repair_pass": rep_pass})

    # Fixture 4: Valid schema → both should accept
    df = pd.DataFrame({"joy": [1], "anger": [0], "fear": [0],
                       "sadness": [0], "surprise": [0]})
    try:
        o = original_brighter_loader(df.copy())
        orig_pass = True
    except Exception:
        orig_pass = False
    try:
        r = repaired_brighter_loader(df.copy())
        rep_pass = True
    except Exception:
        rep_pass = False
    out.append({"fixture": "valid_schema_control",
                "expectation": "accept valid schema",
                "original_pass": orig_pass,
                "repair_pass": rep_pass})

    return out

def main():
    print("=" * 60)
    print("CONTRACT ORACLE FIXTURES (v2 - corrected SC3)")
    print("=" * 60)
    report = {}

    print("\nSC1: Selected-state preservation")
    report["SC1"] = run_sc1()
    for r in report["SC1"]:
        print(f"  {r['fixture']:15s} expected={r['expected_selected']}  orig={r['original_pass']}  repair={r['repair_pass']}")

    print("\nSC2: Validation semantics")
    report["SC2"] = run_sc2()
    for r in report["SC2"]:
        print(f"  {r['fixture']:15s} expected={r['expected_selected']}  orig={r['original_pass']}  repair={r['repair_pass']}")

    print("\nSC3: Schema validation")
    report["SC3"] = run_sc3()
    for r in report["SC3"]:
        print(f"  {r['fixture']:25s} expectation={r['expectation']:30s} orig={r['original_pass']}  repair={r['repair_pass']}")

    orig_pass_count = sum(1 for group in ["SC1", "SC2", "SC3"] for r in report[group] if r["original_pass"])
    rep_pass_count = sum(1 for group in ["SC1", "SC2", "SC3"] for r in report[group] if r["repair_pass"])
    total = sum(len(report[g]) for g in ["SC1", "SC2", "SC3"])

    print(f"\n{'='*60}")
    print(f"Original passes: {orig_pass_count}/{total}")
    print(f"Repair passes:   {rep_pass_count}/{total}")
    print(f"{'='*60}")

    out = os.path.join(REPO_ROOT, "data/repository_analysis/oracle_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
