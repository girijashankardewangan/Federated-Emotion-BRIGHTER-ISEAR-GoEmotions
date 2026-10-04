import os
REPO_ROOT = "/content/Federated-Emotion-BRIGHTER-ISEAR-GoEmotions"
import torch
import copy
import json

class TinyModel(torch.nn.Module):
    """Minimal PyTorch model for state_dict aliasing tests."""
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

def sc1_original_torch(val_seq):
    """Original bug: best_state = model.state_dict() (aliased)."""
    model = TinyModel()
    best_val = -1
    best_state = None
    
    for i, val in enumerate(val_seq):
        # Simulate one training step: update weights
        with torch.no_grad():
            model.fc.weight.fill_(float(i + 1))
            model.fc.bias.fill_(float(i + 1))
        
        # Validation check + selection (buggy)
        if val > best_val:
            best_val = val
            best_state = model.state_dict()  # ← BUG: reference, not copy
    
    # After all training, best_state points to FINAL weights
    return best_state['fc.weight'].flatten()[0].item()

def sc1_repaired_torch(val_seq):
    """Repaired: deepcopy."""
    model = TinyModel()
    best_val = -1
    best_state = None
    
    for i, val in enumerate(val_seq):
        with torch.no_grad():
            model.fc.weight.fill_(float(i + 1))
            model.fc.bias.fill_(float(i + 1))
        
        if val > best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())  # ← FIX
    
    return best_state['fc.weight'].flatten()[0].item()

def run_sc1_pytorch():
    fixtures = [
        ("early_max", [0.9, 0.4, 0.2], 1.0),
        ("later_max", [0.2, 0.8, 0.3], 2.0),
        ("tied_max",  [0.8, 0.8, 0.3], 1.0),
    ]
    out = []
    for name, vals, expected in fixtures:
        orig = sc1_original_torch(vals)
        rep = sc1_repaired_torch(vals)
        out.append({
            "fixture": name,
            "expected_saved_value": expected,
            "original_saved_value": orig,
            "repair_saved_value": rep,
            "original_pass": abs(orig - expected) < 1e-6,
            "repair_pass": abs(rep - expected) < 1e-6,
        })
    return out

def main():
    print("=" * 60)
    print("PYTORCH FIXTURES — SC1 (selected-state preservation)")
    print("=" * 60)
    
    results = run_sc1_pytorch()
    orig_pass = 0
    rep_pass = 0
    for r in results:
        print(f"  {r['fixture']:12s} expected={r['expected_saved_value']:.1f}  "
              f"orig_saved={r['original_saved_value']:.1f} (pass={r['original_pass']})  "
              f"repair_saved={r['repair_saved_value']:.1f} (pass={r['repair_pass']})")
        if r['original_pass']: orig_pass += 1
        if r['repair_pass']: rep_pass += 1
    
    print(f"\nOriginal passes: {orig_pass}/{len(results)}")
    print(f"Repair passes:   {rep_pass}/{len(results)}")
    
    out = os.path.join(REPO_ROOT, "data/repository_analysis/pytorch_fixtures.json")
    with open(out, "w") as f:
        json.dump({"SC1_pytorch": results,
                   "original_pass_count": orig_pass,
                   "repair_pass_count": rep_pass,
                   "total": len(results)}, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
