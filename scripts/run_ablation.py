"""
Ablation study: GMF only vs MLP only vs NeuMF, plus embedding size sweep.
"""

import os, sys
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from data.dataset import get_dataloaders
from models.ncf import GMF, MLP, NeuMF
from evaluate import evaluate_ranking

EPOCHS   = 10
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS  = []

print(f"Device: {DEVICE}")
print("Ladowanie danych...")
train_loader, val_loader, test_loader, n_users, n_items, user_positives = get_dataloaders(
    batch_size=256, n_neg=4
)
print(f"Users: {n_users} | Items: {n_items}\n")


# ── Standalone GMF wrapper ────────────────────────────────────────────────────

class GMFModel(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.gmf = GMF(n_users, n_items, emb_dim)
        self.out  = nn.Linear(emb_dim, 1)
    def forward(self, u, i):
        return torch.sigmoid(self.out(self.gmf(u, i)).squeeze(-1))


class MLPModel(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.mlp = MLP(n_users, n_items, emb_dim, layers=[64, 32, 16])
        self.out  = nn.Linear(16, 1)
    def forward(self, u, i):
        return torch.sigmoid(self.out(self.mlp(u, i)).squeeze(-1))


# ── Training helper ───────────────────────────────────────────────────────────

def train_and_eval(name, model, epochs=EPOCHS, optimizer_cls=optim.Adam, optimizer_kwargs=None):
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer_cls(model.parameters(), lr=0.001, **optimizer_kwargs)
    criterion = nn.BCELoss()
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for users, items, labels in train_loader:
            users, items, labels = users.to(DEVICE), items.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(users, items), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(DEVICE), items.to(DEVICE), labels.to(DEVICE)
                val_loss += criterion(model(users, items), labels).item() * len(labels)
        val_loss /= len(val_loader.dataset)

        marker = " *" if val_loss < best_val else ""
        print(f"  [{name}] Epoch {epoch:2d}/{epochs} | train={train_loss:.4f} val={val_loss:.4f} | {time.time()-t0:.1f}s{marker}")

        if val_loss < best_val:
            best_val = val_loss
            import copy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    metrics = evaluate_ranking(model, test_loader, n_items, user_positives, DEVICE, k=10)
    print(f"  [{name}] >> HitRate@10={metrics['HitRate@10']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}\n")
    return metrics


# ── Ablacja: GMF vs MLP vs NeuMF (emb_dim=32) ────────────────────────────────

print("=" * 55)
print("ABLACJA: GMF vs MLP vs NeuMF  (emb_dim=32)")
print("=" * 55)

for name, model in [
    ("GMF",   GMFModel(n_users, n_items, 32).to(DEVICE)),
    ("MLP",   MLPModel(n_users, n_items, 32).to(DEVICE)),
    ("NeuMF", NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32).to(DEVICE)),
]:
    m = train_and_eval(name, model)
    RESULTS.append({"variant": name, "emb_dim": 32, **m})


# ── Sweep: rozmiary embeddingów (NeuMF) ───────────────────────────────────────

print("=" * 55)
print("SWEEP: rozmiary embeddingów  (NeuMF)")
print("=" * 55)

for emb_dim in [8, 16, 64]:
    name = f"NeuMF-{emb_dim}"
    model = NeuMF(n_users, n_items, gmf_dim=emb_dim, mlp_dim=emb_dim).to(DEVICE)
    m = train_and_eval(name, model)
    RESULTS.append({"variant": name, "emb_dim": emb_dim, **m})


# ── Optimizer: Adam vs AdamW (NeuMF, emb_dim=32) ─────────────────────────────

print("=" * 55)
print("OPTIMIZER: Adam vs AdamW  (NeuMF, emb_dim=32)")
print("=" * 55)

for opt_name, opt_cls, opt_kwargs in [
    ("NeuMF-Adam",  optim.Adam,  {}),
    ("NeuMF-AdamW", optim.AdamW, {"weight_decay": 1e-2}),
]:
    model = NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32).to(DEVICE)
    m = train_and_eval(opt_name, model, optimizer_cls=opt_cls, optimizer_kwargs=opt_kwargs)
    RESULTS.append({"variant": opt_name, "emb_dim": 32, **m})


# ── Podsumowanie ──────────────────────────────────────────────────────────────

print("=" * 55)
print("WYNIKI KONCOWE")
print("=" * 55)
df = pd.DataFrame(RESULTS)[["variant", "emb_dim", "HitRate@10", "NDCG@10"]]
print(df.to_string(index=False))

df.to_csv("results/ablation_results.csv", index=False)
print("\nZapisano: results/ablation_results.csv")
