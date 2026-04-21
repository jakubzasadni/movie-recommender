"""
Neural Matrix Factorization (NeuMF) — He et al., 2017.
arXiv: https://arxiv.org/abs/1708.05031

Architecture:
  GMF branch  : element-wise product of separate user/item embeddings
  MLP branch  : concatenation of separate user/item embeddings → dense layers
  NeuMF       : concat(GMF_out, MLP_out) → Linear(1) → Sigmoid
"""

import torch
import torch.nn as nn


# ── GMF ───────────────────────────────────────────────────────────────────────

class GMF(nn.Module):
    """
    Generalized Matrix Factorization.
    Learns a weighted element-wise product of user and item embeddings.
    The final linear layer (in NeuMF) provides the learned weights.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Returns element-wise product — shape (B, embedding_dim)."""
        return self.user_emb(user_ids) * self.item_emb(item_ids)


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-Layer Perceptron branch.
    Concatenates user and item embeddings, then passes through dense ReLU layers.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if layers is None:
            layers = [64, 32, 16]

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        mlp_layers: list[nn.Module] = []
        in_dim = embedding_dim * 2
        for out_dim in layers:
            mlp_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_dim = layers[-1]

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Returns MLP output — shape (B, layers[-1])."""
        x = torch.cat([self.user_emb(user_ids), self.item_emb(item_ids)], dim=1)
        return self.mlp(x)


# ── NeuMF ─────────────────────────────────────────────────────────────────────

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization.

    Uses *separate* embeddings for GMF and MLP branches (as in the paper),
    allowing each branch to learn independently.

    Args:
        n_users      : number of users
        n_items      : number of items
        gmf_dim      : embedding size for GMF branch
        mlp_dim      : embedding size for MLP branch (input to first dense layer is mlp_dim*2)
        mlp_layers   : hidden layer sizes for MLP branch, e.g. [64, 32, 16]
        dropout      : dropout rate applied after each MLP ReLU
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]

        self.gmf = GMF(n_users, n_items, gmf_dim)
        self.mlp = MLP(n_users, n_items, mlp_dim, mlp_layers, dropout)

        # Prediction layer: concatenated GMF and MLP outputs → scalar logit
        self.predict = nn.Linear(gmf_dim + mlp_layers[-1], 1)
        nn.init.kaiming_uniform_(self.predict.weight, nonlinearity="sigmoid")
        nn.init.zeros_(self.predict.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids : LongTensor (B,)
            item_ids : LongTensor (B,)
        Returns:
            FloatTensor (B,) — predicted interaction probability in [0, 1]
        """
        gmf_out = self.gmf(user_ids, item_ids)        # (B, gmf_dim)
        mlp_out = self.mlp(user_ids, item_ids)        # (B, mlp_layers[-1])
        concat  = torch.cat([gmf_out, mlp_out], dim=1)
        return torch.sigmoid(self.predict(concat).squeeze(-1))
