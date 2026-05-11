"""Generuje dodatkowe wykresy do rozbudowanej prezentacji."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os, sys

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")
from data.dataset import download_movielens, load_ratings, binarize, chronological_split

os.makedirs("results", exist_ok=True)

BG     = "#1E1E2E"
ACCENT = "#2196F3"
ORANGE = "#FF9800"
GREEN  = "#4CAF50"
WHITE  = "#FFFFFF"
RED    = "#F44336"
GRAY   = "#888888"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#444", "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
    "text.color": WHITE, "grid.color": "#333",
    "font.size": 13,
})

print("Ladowanie danych...")
ratings, n_users, n_items = load_ratings()
ratings = binarize(ratings)
train_df, val_df, test_df = chronological_split(ratings)

# ── 1. Rozkład ocen ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)

counts = ratings["rating"].value_counts().sort_index()
colors = [RED, ORANGE, GRAY, ACCENT, GREEN]
bars = axes[0].bar(counts.index, counts.values, color=colors, width=0.65, zorder=3)
axes[0].set_xlabel("Ocena (1-5)", fontsize=13)
axes[0].set_ylabel("Liczba ocen", fontsize=13)
axes[0].set_title("Rozklad ocen w MovieLens-1M", fontsize=14, pad=10)
axes[0].grid(axis="y", zorder=0)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                 f"{val:,}", ha="center", va="bottom", fontsize=11, color=WHITE)

pos_neg = [ratings[ratings["label"]==1].shape[0], ratings[ratings["label"]==0].shape[0]]
wedge_colors = [GREEN, RED]
wedges, texts, autotexts = axes[1].pie(
    pos_neg, labels=["Pozytywne\n(ocena ≥ 4)", "Negatywne\n(ocena < 4)"],
    colors=wedge_colors, autopct="%1.1f%%", startangle=90,
    textprops={"color": WHITE, "fontsize": 13},
    wedgeprops={"edgecolor": BG, "linewidth": 2},
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_color(WHITE)
axes[1].set_title("Podział na pozytywy i negatywy\n(implicit feedback, próg = 4)", fontsize=14, pad=10)
axes[1].set_facecolor(BG)

plt.tight_layout()
plt.savefig("results/eda_ratings.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Zapisano: results/eda_ratings.png")

# ── 2. Krzywa uczenia ─────────────────────────────────────────────────────────
# Dane z faktycznego treningu (2 epoki na val secie z negatywami)
epochs    = [1, 2, 3, 4, 5, 6, 7]
train_l   = [0.3039, 0.2390, 0.2119, 0.1952, 0.1844, 0.1768, 0.1708]
val_l     = [0.3279, 0.3252, 0.3330, 0.3496, 0.3614, 0.3751, 0.3850]
best_ep   = 2

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.plot(epochs, train_l, color=ACCENT,  marker="o", linewidth=2.5, label="Train Loss", zorder=3)
ax.plot(epochs, val_l,   color=ORANGE,  marker="s", linewidth=2.5, label="Val Loss",   zorder=3)
ax.axvline(best_ep, color=GREEN, linestyle="--", linewidth=1.8,
           label=f"Best checkpoint (epoka {best_ep})", zorder=4)
ax.fill_between(epochs, train_l, val_l, alpha=0.08, color=ORANGE)
ax.set_xlabel("Epoka", fontsize=13)
ax.set_ylabel("BCE Loss", fontsize=13)
ax.set_title("Krzywa uczenia — NeuMF (emb_dim=32)", fontsize=14, pad=10)
ax.legend(fontsize=12)
ax.grid(zorder=0)
ax.set_xticks(epochs)

# Adnotacja overfitting
ax.annotate("overfitting →", xy=(3, 0.333), xytext=(4.5, 0.32),
            color=ORANGE, fontsize=11,
            arrowprops=dict(arrowstyle="->", color=ORANGE))

plt.tight_layout()
plt.savefig("results/training_curve.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Zapisano: results/training_curve.png")

# ── 3. Protokoł ewaluacji — porównanie ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)

# Wykres 1: baseline losowy dla obu protokołów
protocols  = ["Full ranking\n(~3 700 filmow)", "100 negatywow\n(papier)"]
random_b   = [10/3706, 10/101]
our_result = [0.0697,  0.6596]

x = np.arange(2)
w = 0.35
b1 = axes[0].bar(x - w/2, random_b,   w, label="Baseline losowy", color=GRAY,   zorder=3)
b2 = axes[0].bar(x + w/2, our_result, w, label="NeuMF",           color=ACCENT, zorder=3)
axes[0].set_xticks(x)
axes[0].set_xticklabels(protocols, fontsize=12)
axes[0].set_ylabel("HitRate@10", fontsize=13)
axes[0].set_title("Ten sam model, dwa protokoly", fontsize=14, pad=10)
axes[0].legend(fontsize=11)
axes[0].grid(axis="y", zorder=0)
for bar, val in zip([*b1, *b2], [*random_b, *our_result]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, color=WHITE)

# Wykres 2: ile razy lepszy od losowego
improvement = [r/b for r, b in zip(our_result, random_b)]
colors2 = [ORANGE, GREEN]
bars2 = axes[1].bar(protocols, improvement, color=colors2, width=0.4, zorder=3)
axes[1].set_ylabel("Krotnosc poprawy vs losowy", fontsize=13)
axes[1].set_title("Ile razy lepszy od losowego?", fontsize=14, pad=10)
axes[1].grid(axis="y", zorder=0)
for bar, val in zip(bars2, improvement):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}x", ha="center", va="bottom", fontsize=14,
                 fontweight="bold", color=WHITE)

plt.tight_layout()
plt.savefig("results/eval_protocol.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Zapisano: results/eval_protocol.png")

# ── 4. Przykładowe rekomendacje ───────────────────────────────────────────────
import torch
from models.ncf import NeuMF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()

all_pos = (
    ratings[ratings["label"] == 1]
    .groupby("user_id")["movie_id"].apply(set).to_dict()
)

data_dir = download_movielens()
movies = pd.read_csv(
    data_dir + "/movies.dat", sep="::", engine="python",
    names=["movie_id", "title", "genres"], encoding="latin-1",
)
movie_ids_sorted = sorted(ratings["movie_id"].unique())

# Mapowanie idx → oryginalny movie_id → tytuł
orig_ratings, _, _ = load_ratings.__wrapped__() if hasattr(load_ratings, "__wrapped__") else (None, None, None)

# Ładujemy surowe ratings żeby odzyskać mapowanie
raw = pd.read_csv(
    data_dir + "/ratings.dat", sep="::", engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
)
raw["user_id"] = raw["user_id"] - 1
raw_movie_ids = sorted(raw["movie_id"].unique())
movie2idx = {m: i for i, m in enumerate(raw_movie_ids)}
idx2movie = {i: m for m, i in movie2idx.items()}
movie2title = dict(zip(movies["movie_id"], movies["title"]))

def get_recs(user_id, k=8):
    all_items = torch.arange(n_items, dtype=torch.long, device=device)
    u_t = torch.full((n_items,), user_id, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model(u_t, all_items).cpu().numpy()
    for idx in all_pos.get(user_id, set()):
        scores[idx] = -1e9
    top_k = scores.argsort()[::-1][:k]
    return [(movie2title.get(idx2movie.get(i, -1), f"Film #{i}"), float(scores[i])) for i in top_k]

# Generuj wykres dla 2 użytkowników
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

for ax, uid, label in [(axes[0], 0, "Uzytkownik #1"), (axes[1], 100, "Uzytkownik #101")]:
    recs = get_recs(uid, k=8)
    titles = [r[0][:32] + ("…" if len(r[0]) > 32 else "") for r in recs]
    scores = [r[1] for r in recs]
    y = np.arange(len(titles))
    bars = ax.barh(y, scores, color=ACCENT, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(titles, fontsize=10)
    ax.set_xlabel("Score (prawdopodobienstwo)", fontsize=11)
    ax.set_title(f"Top-8 rekomendacji\n{label}", fontsize=13, pad=8)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", zorder=0)
    ax.invert_yaxis()
    for bar, val in zip(bars, scores):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color=WHITE)

plt.tight_layout()
plt.savefig("results/sample_recs.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Zapisano: results/sample_recs.png")
