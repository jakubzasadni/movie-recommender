"""
Evaluation script: HitRate@10, NDCG@10 for NeuMF on MovieLens-1M test set.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pt --k 10
"""

import argparse
import math

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.ncf import NeuMF


# ── Ranking metrics ───────────────────────────────────────────────────────────

def hit_rate_at_k(ranked: list[int], relevant: set[int], k: int = 10) -> float:
    """1 if any item in top-k is relevant, else 0."""
    return float(bool(set(ranked[:k]) & relevant))


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @ k."""
    dcg = sum(
        1.0 / math.log2(rank + 2)          # rank is 0-based → log2(rank+2)
        for rank, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ── Evaluation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ranking(
    model: torch.nn.Module,
    test_loader,
    n_items: int,
    user_positives: dict,
    device: torch.device,
    k: int = 10,
) -> dict:
    """
    For each user in test set:
      1. Score all items with the model.
      2. Mask out all known-positive items except the current test item.
      3. Rank and compute HitRate@k and NDCG@k.
    """
    model.eval()
    all_items = torch.arange(n_items, dtype=torch.long, device=device)

    hit_rates, ndcgs = [], []
    seen_users: set[int] = set()

    for users, items, _ in tqdm(test_loader, desc=f"Evaluating (k={k})"):
        for u, pos_item in zip(users.tolist(), items.tolist()):
            if u in seen_users:
                continue
            seen_users.add(u)

            user_tensor = torch.full((n_items,), u, dtype=torch.long, device=device)
            scores = model(user_tensor, all_items).cpu().numpy()

            # Mask known positives (except the one being evaluated)
            for idx in user_positives.get(u, set()) - {pos_item}:
                scores[idx] = -np.inf

            ranked = np.argsort(scores)[::-1][:k].tolist()
            hit_rates.append(hit_rate_at_k(ranked, {pos_item}, k))
            ndcgs.append(ndcg_at_k(ranked, {pos_item}, k))

    return {
        f"HitRate@{k}": float(np.mean(hit_rates)),
        f"NDCG@{k}":    float(np.mean(ndcgs)),
        "n_users":      len(seen_users),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    _, _, test_loader, n_users, n_items, user_positives = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_neg=0,
    )

    mlp_layers = [int(x) for x in args.mlp_layers.split(",")]
    model = NeuMF(
        n_users=n_users,
        n_items=n_items,
        gmf_dim=args.emb_dim,
        mlp_dim=args.emb_dim,
        mlp_layers=mlp_layers,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded: {args.checkpoint}")

    metrics = evaluate_ranking(model, test_loader, n_items, user_positives, device, k=args.k)

    print("\n--- Test Results ---")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    # Print vs paper targets
    print("\n--- vs. Paper Targets (He et al. 2017, ML-1M) ---")
    print(f"  HitRate@10 target: > 0.65  (got {metrics.get('HitRate@10', 0):.4f})")
    print(f"  NDCG@10    target: > 0.38  (got {metrics.get('NDCG@10', 0):.4f})")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NeuMF on MovieLens-1M test set")
    p.add_argument("--checkpoint", type=str,   default="checkpoints/best_model.pt")
    p.add_argument("--data_dir",   type=str,   default=None)
    p.add_argument("--emb_dim",    type=int,   default=32)
    p.add_argument("--mlp_layers", type=str,   default="64,32,16")
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--k",          type=int,   default=10)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
