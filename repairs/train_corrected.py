# Candidate contract repair.
#
# Historical train.py is intentionally preserved.
# Historical CSV results are intentionally preserved.
# This file is NOT regenerated historical evidence.
#
# The candidate repair addresses the mutable checkpoint
# state issue identified in the audit.

import copy

 # ============================================================
# BRIGHTER + ISEAR FEDERATED EMOTION CLASSIFICATION
# Centralized vs Federated DistilBERT Comparison
# ============================================================

import os, json, random, shutil, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BRIGHTER_SEEDS = list(range(10))   # 0..9
ISEAR_SEEDS = list(range(5))       # 0..4
GOEMOTIONS_SEEDS = list(range(5))   # 0..4
METHODS = ['C1', 'F1', 'F2', 'F3']

BATCH_SIZE = 8
MAX_LEN = 64
LR = 1e-5
EPOCHS = 5
ROUNDS = 5
CLIENTS = 5
DIRICHLET_ALPHA = 0.5
CLIP_NORM = 1.0
NOISE_MULT = 1.1

LABELS = ['joy', 'anger', 'fear', 'sadness', 'surprise']

SAVE_DIR = '/content/drive/MyDrive/BRIGHTER_results'
os.makedirs(SAVE_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(SAVE_DIR, 'results.csv')
PROGRESS_JSON = os.path.join(SAVE_DIR, 'progress.json')


# ------------------------------------------------------------
# DATASET LOADING
# ------------------------------------------------------------
def load_brighter():
    """Load BRIGHTER English subset from Hugging Face."""
    dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "eng")
    train = dataset['train'].to_pandas()
    val = dataset['dev'].to_pandas()
    test = dataset['test'].to_pandas()
    for df in [train, val, test]:
        for lbl in LABELS:
            if lbl not in df.columns:
                df[lbl] = 0
    return train, val, test


def load_isear():
    """Load ISEAR dataset (pipe-delimited, numeric emotion codes)."""
    url = "https://raw.githubusercontent.com/bdotloh/isear_dataset/master/isear.csv"
    df = pd.read_csv(url, sep='|', on_bad_lines='skip', engine='python')
    print(f"ISEAR loaded: {df.shape}")

    # Use SIT column for text and EMOT for numeric emotion code
    df = df[['SIT', 'EMOT']].rename(columns={'SIT': 'text', 'EMOT': 'emotion'})

    # ISEAR codes: 1=joy, 2=fear, 3=anger, 4=sadness, 5=disgust, 6=shame, 7=guilt
    code_to_emotion = {
        1: 'joy', 2: 'fear', 3: 'anger', 4: 'sadness',
        5: None, 6: None, 7: None
    }
    df['emotion'] = df['emotion'].map(code_to_emotion)
    df = df.dropna(subset=['emotion'])
    print(f"ISEAR after filtering: {len(df)} rows")

    for lbl in LABELS:
        df[lbl] = (df['emotion'] == lbl).astype(int)
    df = df[['text'] + LABELS]

    np.random.seed(42)
    idx = np.random.permutation(len(df))
    train_end = int(0.7 * len(df))
    val_end = int(0.85 * len(df))
    train = df.iloc[idx[:train_end]].reset_index(drop=True)
    val = df.iloc[idx[train_end:val_end]].reset_index(drop=True)
    test = df.iloc[idx[val_end:]].reset_index(drop=True)
    print(f"ISEAR split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test




# ------------------------------------------------------------
# GOEMOTIONS LOADER
# ------------------------------------------------------------
def load_goemotions():
    """Load GoEmotions with proper 5 label columns."""
    from datasets import load_dataset
    import pandas as pd

    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    target_labels = ["joy", "anger", "fear", "sadness", "surprise"]
    label_names = dataset["train"].features["labels"].feature.names
    target_indices = [label_names.index(l) for l in target_labels]

    def convert(example):
        multihot = [0] * 5
        for lbl in example["labels"]:
            if lbl in target_indices:
                multihot[target_indices.index(lbl)] = 1
        return {"multihot": multihot}

    dataset = dataset.map(convert)

    def to_df(split):
        df = dataset[split].to_pandas()
        multihot_df = pd.DataFrame(
            df["multihot"].tolist(),
            columns=target_labels,
            index=df.index
        )
        result = pd.concat([df[["text"]], multihot_df], axis=1)
        result = result[result[target_labels].sum(axis=1) > 0].reset_index(drop=True)
        return result

    train = to_df("train")
    val = to_df("validation")
    test = to_df("test")
    print(f"GoEmotions: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Columns: {train.columns.tolist()}")
    return train, val, test



class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ------------------------------------------------------------
# CENTRALIZED TRAINING (C1)
# ------------------------------------------------------------
def train_centralized(train_df, val_df, test_df, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(LABELS),
        problem_type='multi_label_classification'
    ).to(DEVICE)

    train_ds = EmotionDataset(train_df['text'].tolist(), train_df[LABELS].values, tokenizer)
    val_ds = EmotionDataset(val_df['text'].tolist(), val_df[LABELS].values, tokenizer)
    test_ds = EmotionDataset(test_df['text'].tolist(), test_df[LABELS].values, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)

    best_val_f1 = -1
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                preds.append(torch.sigmoid(outputs.logits).cpu().numpy())
                trues.append(batch['labels'].cpu().numpy())
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        pred_bin = (preds > 0.5).astype(int)
        val_macro = f1_score(trues, pred_bin, average='macro', zero_division=0)
        if val_macro > best_val_f1:
            best_val_f1 = val_macro
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    # Test
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds.append(torch.sigmoid(outputs.logits).cpu().numpy())
            trues.append(batch['labels'].cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    pred_bin = (preds > 0.5).astype(int)
    macro = f1_score(trues, pred_bin, average='macro', zero_division=0)
    micro = f1_score(trues, pred_bin, average='micro', zero_division=0)
    exact = accuracy_score(trues, pred_bin)
    return macro, micro, exact


# ------------------------------------------------------------
# FEDERATED TRAINING (F1, F2, F3)
# ------------------------------------------------------------
def train_federated(train_df, val_df, test_df, seed, method):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    global_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(LABELS),
        problem_type='multi_label_classification'
    ).to(DEVICE)

    # Distribute samples ensuring min 2 per client
    n_samples = len(train_df)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    max_possible_clients = max(1, n_samples // 2)
    actual_K = min(CLIENTS, max_possible_clients)

    proportions = np.random.dirichlet([DIRICHLET_ALPHA] * actual_K)
    proportions = (proportions * n_samples).astype(int)

    for i in range(actual_K):
        if proportions[i] < 2:
            proportions[i] = 2

    diff = n_samples - proportions.sum()
    if diff != 0:
        proportions[np.argmax(proportions)] += diff

    client_data = []
    start = 0
    for i in range(actual_K):
        end = min(start + proportions[i], n_samples)
        client_data.append(train_df.iloc[indices[start:end]].reset_index(drop=True))
        start = end

    client_data = [c for c in client_data if len(c) >= 2]
    actual_clients = len(client_data)
    print(f"  Federated partition: {actual_clients} clients, sizes={[len(c) for c in client_data]}")

    # Federated rounds
    for rnd in range(ROUNDS):
        local_states = []
        for cid in range(actual_clients):
            local_model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', num_labels=len(LABELS),
                problem_type='multi_label_classification'
            ).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())

            train_ds = EmotionDataset(client_data[cid]['text'].tolist(), client_data[cid][LABELS].values, tokenizer)
            bs = min(BATCH_SIZE, len(client_data[cid]))
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

            optimizer = torch.optim.SGD(local_model.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)
            local_model.train()
            for batch in train_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = local_model(**batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute update
            update = {}
            global_state = global_model.state_dict()
            for name, param in local_model.named_parameters():
                update[name] = param.data - global_state[name].data

            # Clipping (F2, F3)
            if method in ['F2', 'F3']:
                total_norm = torch.sqrt(sum(torch.sum(v**2) for v in update.values()))
                clip_coef = CLIP_NORM / (total_norm + 1e-6)
                if clip_coef < 1:
                    for name in update:
                        update[name] *= clip_coef

            # Noise (F3)
            if method == 'F3':
                for name in update:
                    noise = torch.randn_like(update[name]) * NOISE_MULT * CLIP_NORM
                    update[name] += noise

            local_states.append(update)

        # Aggregate
        total_samples = sum(len(d) for d in client_data)
        new_state = global_model.state_dict().copy()
        for name in new_state:
            new_state[name] = sum(
                (len(client_data[i]) / total_samples) * (global_model.state_dict()[name] + local_states[i][name])
                for i in range(actual_clients)
            )
        global_model.load_state_dict(new_state)

    # Test evaluation
    test_ds = EmotionDataset(test_df['text'].tolist(), test_df[LABELS].values, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    global_model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = global_model(**batch)
            preds.append(torch.sigmoid(outputs.logits).cpu().numpy())
            trues.append(batch['labels'].cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    pred_bin = (preds > 0.5).astype(int)
    macro = f1_score(trues, pred_bin, average='macro', zero_division=0)
    micro = f1_score(trues, pred_bin, average='micro', zero_division=0)
    exact = accuracy_score(trues, pred_bin)
    return macro, micro, exact


# ------------------------------------------------------------
# PROGRESS AND CHECKPOINT UTILITIES
# ------------------------------------------------------------
def load_progress():
    if os.path.exists(PROGRESS_JSON):
        with open(PROGRESS_JSON) as f:
            return json.load(f)
    return {"completed": []}


def save_progress(progress):
    with open(PROGRESS_JSON, 'w') as f:
        json.dump(progress, f)


def append_result(row):
    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_CSV, index=False)


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
def main():
    progress = load_progress()
    completed = set(tuple(x) for x in progress['completed'])

    print("Loading BRIGHTER...")
    br_train, br_val, br_test = load_brighter()

    print("Loading ISEAR...")
    is_train, is_val, is_test = load_isear()

    print("Loading GoEmotions...")
    go_train, go_val, go_test = load_goemotions()

    tasks = []
    for seed in BRIGHTER_SEEDS:
        for method in METHODS:
            tasks.append(('BRIGHTER', method, seed, br_train, br_val, br_test))
    for seed in ISEAR_SEEDS:
        for method in METHODS:
            tasks.append(('ISEAR', method, seed, is_train, is_val, is_test))
    for seed in GOEMOTIONS_SEEDS:
        for method in METHODS:
            tasks.append(('GoEmotions', method, seed, go_train, go_val, go_test))

    for dataset, method, seed, train_df, val_df, test_df in tasks:
        key = (dataset, method, seed)
        if key in completed:
            print(f"Skipping {key} (already done)")
            continue

        print(f"Running {dataset} | {method} | seed {seed}")
        if method == 'C1':
            macro, micro, exact = train_centralized(train_df, val_df, test_df, seed)
        else:
            macro, micro, exact = train_federated(train_df, val_df, test_df, seed, method)

        row = {
            'dataset': dataset, 'method': method, 'seed': seed,
            'macro_f1': macro, 'micro_f1': micro, 'exact_match': exact
        }
        append_result(row)
        completed.add(key)
        progress['completed'] = list(completed)
        save_progress(progress)
        print(f"Result: macro={macro:.4f}, micro={micro:.4f}, exact={exact:.4f}")

    print("All runs completed!")


if __name__ == '__main__':
    main()