import os
REPO_ROOT = "/content/Federated-Emotion-BRIGHTER-ISEAR-GoEmotions"
import difflib

REPAIRS = [
    {
        "anchor": "best_state = model.state_dict()",
        "replacement": "import copy\n            best_state = copy.deepcopy(model.state_dict())",
    },
]

def main():
    src = os.path.join(REPO_ROOT, "train.py")
    with open(src) as f:
        original = f.read()
    repaired = original
    for r in REPAIRS:
        if r["anchor"] in repaired:
            repaired = repaired.replace(r["anchor"], r["replacement"], 1)
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        repaired.splitlines(keepends=True),
        fromfile="a/train.py", tofile="b/train.py",
    )
    patch = "".join(diff)
    os.makedirs(os.path.join(REPO_ROOT, "repairs"), exist_ok=True)
    out = os.path.join(REPO_ROOT, "repairs/train_contracts.patch")
    with open(out, "w") as f:
        f.write(patch)
    print(f"Patch: {out} ({len(patch.splitlines())} lines)")

if __name__ == "__main__":
    main()
