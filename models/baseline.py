"""
Baseline models for MovieLens-1M recommendation.

  PopularityBaseline  — recommends most-interacted items globally
  MatrixFactorization — explicit-feedback MF (dot-product + bias), trained with MSE
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── Popularity baseline ───────────────────────────────────────────────────────

class PopularityBaseline:
    """
    Recommends the globally most popular items (by positive-interaction count).
    Used as the simplest non-personalized baseline.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.popular_items: list[int] = []

    def fit(self, train_df: pd.DataFrame) -> "PopularityBaseline":
        counts = train_df[train_df["label"] == 1]["movie_id"].value_counts()
        self.popular_items = counts.index.tolist()
        return self

    def recommend(self, user_id: int, exclude: set | None = None, k: int | None = None) -> list[int]:
        """Return top-k items, excluding any already seen by the user."""
        k = k or self.top_k
        recs = []
        for item in self.popular_items:
            if exclude and item in exclude:
                continue
            recs.append(item)
            if len(recs) == k:
                break
        return recs


# ── Matrix Factorization (explicit feedback) ──────────────────────────────────

class MatrixFactorization(nn.Module):
    """
    Explicit-feedback MF: predicts a rating as dot(u, i) + bias_u + bias_i + global_bias.
    Train with MSELoss on raw 1-5 ratings to get RMSE baseline.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, embedding_dim)
        self.item_emb  = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight,  std=0.01)
        nn.init.normal_(self.item_emb.weight,  std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u    = self.user_emb(user_ids)
        i    = self.item_emb(item_ids)
        dot  = (u * i).sum(dim=1)
        bias = (
            self.user_bias(user_ids).squeeze(-1)
            + self.item_bias(item_ids).squeeze(-1)
            + self.global_bias
        )
        return dot + bias
