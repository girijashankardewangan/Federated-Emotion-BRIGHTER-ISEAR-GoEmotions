import os
REPO_ROOT = "/content/Federated-Emotion-BRIGHTER-ISEAR-GoEmotions"

import zipfile
import pandas as pd
import json
import hashlib
import ast

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def check_f1_f2_equality(df):
    pairs = []
    method_col = "method" if "method" in df.columns else "Method"
    metric_cols = [c for c in ["macro_f1", "micro_f1", "exact_match", "Macro F1", "Micro F1", "Exact Match"] if c in df.columns]
    for (dataset, seed), group in df.groupby(["dataset", "seed"]):
        f1 = group[group[method_col] == "F1"]
        f2 = group[group[method_col] == "F2"]
        if len(f1) == 1 and len(f2) == 1:
            for metric in metric_cols:
                equal = abs(float(f1[metric].values[0]) - float(f2[metric].values[0])) < 1e-6
                pairs.append({"dataset": dataset, "seed": int(seed), "metric": metric, "equal": equal})
    return pairs

def check_f3_zeros(df):
    method_col = "method" if "method" in df.columns else "Method"
    f3 = df[df[method_col] == "F3"]
    zero_count = 0
    for _, row in f3.iterrows():
        m = float(row.get("macro_f1", row.get("Macro F1", 0)))
        if abs(m) < 1e-9:
            zero_count += 1
    return {"total_f3": len(f3), "zero_macro": zero_count}

def ast_check_source(source_path):
    findings = {}
    with open(source_path) as f:
        source = f.read()
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "best_state":
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "state_dict":
                            findings["SC1"] = {"line": node.lineno, "issue": "state_dict without deepcopy", "confirmed": True}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train_federated":
            arg_names = [a.arg for a in node.args.args]
            uses_val = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == "val_df" and sub is not node:
                    uses_val = True
                    break
            findings["SC2"] = {
                "accepts_val_df": "val_df" in arg_names,
                "uses_val_df": uses_val,
            }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "LABELS":
                    if isinstance(node.value, ast.List):
                        labels = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
                        findings["SC3"] = {"labels": labels, "has_surprise": "surprise" in labels}
    return findings

def main():
    print("=" * 60)
    print("SOFTWARE CONTRACT VERIFICATION")
    print("=" * 60)
    report = {"files": {}, "checks": {}}
    
    for name, path in [
        ("results.csv", os.path.join(REPO_ROOT, "results.csv")),
        ("goemotions_results.csv", os.path.join(REPO_ROOT, "goemotions_results.csv")),
        ("train.py", os.path.join(REPO_ROOT, "train.py")),
    ]:
        if os.path.exists(path):
            report["files"][name] = {"sha256": sha256_file(path), "size": os.path.getsize(path)}
            print(f"OK {name}: {report['files'][name]['sha256'][:16]}...")
    
    r = pd.read_csv(os.path.join(REPO_ROOT, "results.csv"))
    g = pd.read_csv(os.path.join(REPO_ROOT, "goemotions_results.csv"))
    df = pd.concat([r, g], ignore_index=True)
    print(f"Total rows: {len(df)} (expected 80)")
    print(f"Columns: {list(df.columns)}")
    report["checks"]["grid_rows"] = len(df)
    
    print("\nSC5: F1/F2 equality")
    pairs = check_f1_f2_equality(df)
    equal = sum(1 for p in pairs if p["equal"])
    print(f"F1/F2 equal: {equal}/{len(pairs)}")
    report["checks"]["f1_f2_equal"] = equal
    report["checks"]["f1_f2_total"] = len(pairs)
    
    print("\nSC6: F3 zeros")
    f3 = check_f3_zeros(df)
    print(f"F3 zero macro: {f3['zero_macro']}/{f3['total_f3']}")
    report["checks"]["f3"] = f3
    
    print("\nSC1/SC2/SC3: AST checks")
    ast_findings = ast_check_source(os.path.join(REPO_ROOT, "train.py"))
    for k, v in ast_findings.items():
        print(f"  {k}: {v}")
    report["checks"]["ast"] = ast_findings
    
    out = os.path.join(REPO_ROOT, "data/repository_analysis/software_contract_checks.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
