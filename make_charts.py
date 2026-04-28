"""Generuje wykresy do prezentacji na podstawie wyników."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("results", exist_ok=True)

BG     = "#1E1E2E"
ACCENT = "#2196F3"
ORANGE = "#FF9800"
GREEN  = "#4CAF50"
GRAY   = "#AAAAAA"
WHITE  = "#FFFFFF"
RED    = "#F44336"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#444", "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
    "text.color": WHITE, "grid.color": "#333333",
    "font.size": 13,
})

df = pd.read_csv("results/ablation_results.csv")

# ── 1. Ablacja: GMF vs MLP vs NeuMF ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

ablation = df[df["variant"].isin(["GMF", "MLP", "NeuMF"])]
x = np.arange(len(ablation))
colors = [ACCENT, ORANGE, GREEN]

for ax, metric, target, label in [
    (axes[0], "HitRate@10", 0.65, "HitRate@10"),
    (axes[1], "NDCG@10",    0.38, "NDCG@10"),
]:
    bars = ax.bar(x, ablation[metric], color=colors, width=0.5, zorder=3)
    ax.axhline(target, color=RED, linestyle="--", linewidth=1.5,
               label=f"Cel z papieru ({target})", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation["variant"], fontsize=14)
    ax.set_ylabel(label, fontsize=13)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=11)
    for bar, val in zip(bars, ablation[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12, color=WHITE)

fig.suptitle("Ablacja: GMF vs MLP vs NeuMF  (emb_dim=32, 10 epok)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("results/ablation_components.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Zapisano: results/ablation_components.png")

# ── 2. Sweep embeddingów ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

sweep = df[df["variant"].str.startswith("NeuMF")].copy()
sweep["emb_dim"] = sweep["emb_dim"].astype(int)
sweep = sweep.sort_values("emb_dim")
xs = np.arange(len(sweep))
sweep_colors = [GREEN, GREEN, ACCENT, ORANGE]

for ax, metric, target in [
    (axes[0], "HitRate@10", 0.65),
    (axes[1], "NDCG@10",    0.38),
]:
    bars = ax.bar(xs, sweep[metric], color=sweep_colors, width=0.5, zorder=3)
    ax.axhline(target, color=RED, linestyle="--", linewidth=1.5,
               label=f"Cel z papieru ({target})", zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in sweep["emb_dim"]], fontsize=14)
    ax.set_xlabel("emb_dim", fontsize=13)
    ax.set_ylabel(metric, fontsize=13)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=11)
    for bar, val in zip(bars, sweep[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12, color=WHITE)

fig.suptitle("Wpływ rozmiaru embeddingów — NeuMF  (10 epok)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("results/ablation_embeddings.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Zapisano: results/ablation_embeddings.png")

# ── 3. Pełna tabela wyników (jako wykres) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor(BG)
ax.axis("off")

rows_data = [
    ["GMF",       "32", "0.6582", "0.3753"],
    ["MLP",       "32", "0.6366", "0.3616"],
    ["NeuMF",     "32", "0.6534", "0.3745"],
    ["NeuMF-8",   "8",  "0.6560", "0.3756"],
    ["NeuMF-16",  "16", "0.6562", "0.3754"],
    ["NeuMF-64",  "64", "0.6408", "0.3649"],
]
col_labels = ["Model", "emb_dim", "HitRate@10", "NDCG@10"]

table = ax.table(
    cellText=rows_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 2.2)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor("#2A2A3E" if row % 2 == 0 else "#1E1E2E")
    cell.set_edgecolor("#444")
    cell.set_text_props(color=WHITE)
    if row == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color=WHITE, fontweight="bold")

plt.tight_layout()
plt.savefig("results/full_results_table.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Zapisano: results/full_results_table.png")
