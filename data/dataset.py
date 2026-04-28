"""
MovieLens-1M dataset utilities.

Pipeline:
  download → load → binarize (rating ≥ 4 → positive) →
  chronological split (80/10/10) → negative sampling → DataLoader
"""

import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
_DEFAULT_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ── Download ─────────────────────────────────────────────────────────────────

def download_movielens(raw_dir: str = _DEFAULT_RAW) -> str:
    """Download and extract MovieLens-1M. Returns path to ml-1m/ directory."""
    raw_dir = os.path.abspath(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    extract_path = os.path.join(raw_dir, "ml-1m")

    if not os.path.exists(extract_path):
        zip_path = os.path.join(raw_dir, "ml-1m.zip")
        print(f"Downloading MovieLens-1M to {zip_path} ...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        os.remove(zip_path)
        print("Download complete.")

    return extract_path


# ── Load ─────────────────────────────────────────────────────────────────────

def load_ratings(data_dir: str = None) -> tuple[pd.DataFrame, int, int]:
    """
    Load ratings.dat; re-index users and movies to 0-based contiguous ints.

    Returns:
        ratings DataFrame, n_users, n_items
    """
    if data_dir is None:
        data_dir = download_movielens()

    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    # Re-index to 0-based contiguous integers
    df["user_id"] = df["user_id"] - 1  # original IDs are 1-indexed

    movie_ids = sorted(df["movie_id"].unique())
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    df["movie_id"] = df["movie_id"].map(movie2idx)

    n_users = df["user_id"].nunique()
    n_items = len(movie_ids)
    return df, n_users, n_items


# ── Binarize ─────────────────────────────────────────────────────────────────

def binarize(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Convert explicit ratings to implicit feedback: 1 if rating ≥ threshold."""
    out = df.copy()
    out["label"] = (out["rating"] >= threshold).astype(int)
    return out


# ── Split ─────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-user chronological split:
      oldest 80% → train, next 10% → val, newest 10% → test.
    Guarantees at least 1 interaction per user in val and test.
    """
    df = df.sort_values(["user_id", "timestamp"])
    train_l, val_l, test_l = [], [], []

    for _, grp in df.groupby("user_id"):
        n = len(grp)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        train_l.append(grp.iloc[: n - n_test - n_val])
        val_l.append(grp.iloc[n - n_test - n_val : n - n_test])
        test_l.append(grp.iloc[n - n_test :])

    return (
        pd.concat(train_l).reset_index(drop=True),
        pd.concat(val_l).reset_index(drop=True),
        pd.concat(test_l).reset_index(drop=True),
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for implicit-feedback NCF training.

    Training mode:  for each positive (user, item), samples `n_neg` negatives.
    Eval mode:      returns positive pairs only (ranking is done externally).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_items: int,
        n_neg: int = 4,
        is_training: bool = True,
        all_positive: dict | None = None,
        seed: int = 42,
    ):
        self.n_items = n_items
        self.n_neg = n_neg
        self.is_training = is_training

        pos_df = df[df["label"] == 1]
        self.users = pos_df["user_id"].values
        self.items = pos_df["movie_id"].values

        # Known positives per user — used to avoid false negatives when sampling
        if all_positive is not None:
            self.user_positives = all_positive
        else:
            self.user_positives = (
                df[df["label"] == 1]
                .groupby("user_id")["movie_id"]
                .apply(set)
                .to_dict()
            )

        if is_training:
            self._build_samples(seed)

    def _build_samples(self, seed: int):
        rng = np.random.default_rng(seed)
        u_list, i_list, l_list = [], [], []

        for u, i in zip(self.users, self.items):
            u_list.append(u); i_list.append(i); l_list.append(1)
            pos_set = self.user_positives.get(u, set())
            negs = set()
            while len(negs) < self.n_neg:
                neg = int(rng.integers(0, self.n_items))
                if neg not in pos_set:
                    negs.add(neg)
            for neg in negs:
                u_list.append(u); i_list.append(neg); l_list.append(0)

        self._users = np.array(u_list)
        self._items = np.array(i_list)
        self._labels = np.array(l_list, dtype=np.float32)

    def __len__(self):
        return len(self._users) if self.is_training else len(self.users)

    def __getitem__(self, idx):
        if self.is_training:
            return (
                torch.tensor(self._users[idx], dtype=torch.long),
                torch.tensor(self._items[idx], dtype=torch.long),
                torch.tensor(self._labels[idx], dtype=torch.float32),
            )
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(1.0),
        )


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str = None,
    batch_size: int = 256,
    n_neg: int = 4,
    num_workers: int = 0,
) -> tuple:
    """
    Full pipeline: download → load → binarize → split → DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, n_users, n_items, user_positives
    """
    ratings, n_users, n_items = load_ratings(data_dir)
    ratings = binarize(ratings)
    train_df, val_df, test_df = chronological_split(ratings)

    all_pos = (
        ratings[ratings["label"] == 1]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    train_ds = MovieLensDataset(train_df, n_items, n_neg=n_neg, is_training=True,  all_positive=all_pos)
    val_ds   = MovieLensDataset(val_df,   n_items, n_neg=n_neg, is_training=True,  all_positive=all_pos)
    test_ds  = MovieLensDataset(test_df,  n_items, n_neg=0,     is_training=False, all_positive=all_pos)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, n_users, n_items, all_pos
