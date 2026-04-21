"""
Training script for NeuMF on MovieLens-1M.

Usage:
    python train.py
    python train.py --emb_dim 64 --epochs 30 --lr 0.0005
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import get_dataloaders
from models.ncf import NeuMF


# ── Train / eval helpers ──────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for users, items, labels in loader:
        users  = users.to(device)
        items  = items.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(users, items), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for users, items, labels in loader:
        users  = users.to(device)
        items  = items.to(device)
        labels = labels.to(device)
        total_loss += criterion(model(users, items), labels).item() * len(labels)
    return total_loss / len(loader.dataset)


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_loader, val_loader, _, n_users, n_items, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_neg=args.n_neg,
    )
    print(f"  Users: {n_users} | Items: {n_items}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    mlp_layers = [int(x) for x in args.mlp_layers.split(",")]
    model = NeuMF(
        n_users=n_users,
        n_items=n_items,
        gmf_dim=args.emb_dim,
        mlp_dim=args.emb_dim,
        mlp_layers=mlp_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nEpoch | Train Loss | Val Loss  | Time")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_loss(model, val_loader, criterion, device)
        elapsed    = time.time() - t0

        marker = " *" if val_loss < best_val_loss else ""
        print(f"  {epoch:3d}  | {train_loss:.4f}     | {val_loss:.4f}    | {elapsed:.1f}s{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {args.save_dir}/best_model.pt")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train NeuMF on MovieLens-1M")
    p.add_argument("--data_dir",   type=str,   default=None,        help="Path to ml-1m/ directory (auto-download if omitted)")
    p.add_argument("--save_dir",   type=str,   default="checkpoints")
    p.add_argument("--emb_dim",    type=int,   default=32,           help="Embedding size for GMF and MLP branches")
    p.add_argument("--mlp_layers", type=str,   default="64,32,16",   help="MLP hidden layer sizes, comma-separated")
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--lr",         type=float, default=0.001)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--n_neg",      type=int,   default=4,            help="Negative samples per positive")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--patience",   type=int,   default=5,            help="Early stopping patience")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
