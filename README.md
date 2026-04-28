# System Rekomendacji Filmów — NeuMF

Implementacja algorytmu **Neural Matrix Factorization (NeuMF)** na datasecie **MovieLens-1M** w PyTorch.

Projekt akademicki na podstawie: *He et al., 2017 — Neural Collaborative Filtering* ([arXiv:1708.05031](https://arxiv.org/abs/1708.05031))

---

## Co robi ten projekt

Model uczy się przewidywać, czy użytkownik polubi dany film, na podstawie historii jego ocen. Zamiast klasycznego filtrowania kolaboratywnego (zwykłe mnożenie wektorów) używamy sieci neuronowej, która wychwytuje nieliniowe zależności między preferencjami użytkowników a filmami.

**Dane wejściowe:** `user_id` + `movie_id`  
**Dane wyjściowe:** prawdopodobieństwo 0–1 (czy użytkownik polubi film)

---

## Struktura projektu

```
movie-recommender/
├── data/
│   └── dataset.py        # pobieranie ML-1M, przetwarzanie, DataLoader
├── models/
│   ├── ncf.py            # GMF + MLP + NeuMF
│   └── baseline.py       # PopularityBaseline, MatrixFactorization
├── notebooks/
│   ├── ablation.ipynb    # GMF vs MLP vs NeuMF, rozmiary embeddingów
│   └── results.ipynb     # wykresy wyników, przykładowe rekomendacje
├── train.py              # trening modelu
├── evaluate.py           # HitRate@10, NDCG@10
└── requirements.txt
```

---

## Szybki start

```bash
pip install -r requirements.txt

# Trening (ML-1M zostanie pobrany automatycznie przy pierwszym uruchomieniu)
python train.py

# Ewaluacja wytrenowanego modelu
python evaluate.py
```

Opcje treningu:
```bash
python train.py --emb_dim 64 --mlp_layers 128,64,32 --epochs 30 --lr 0.0005
```

---

## Dataset — MovieLens-1M

| Parametr        | Wartość              |
|-----------------|----------------------|
| Użytkownicy     | 6 040                |
| Filmy           | ~3 900               |
| Oceny           | 1 000 209            |
| Skala ocen      | 1–5                  |
| Wypełnienie     | ~4% (rzadka macierz) |

Oceny są **binaryzowane**: ≥ 4 → pozytywna interakcja (użytkownik lubi film).  
Podział: **80% trening / 10% walidacja / 10% test** (chronologicznie per użytkownik).

---

## Architektura modelu

```
user_id ──► [GMF embedding] ──► element-wise product ──┐
item_id ──► [GMF embedding] ──────────────────────────►├──► concat ──► Linear(1) ──► Sigmoid
                                                        │
user_id ──► [MLP embedding] ──► concat ──► MLP ────────┘
item_id ──► [MLP embedding] ──►
```

**GMF** wychwytuje liniowe zależności między użytkownikiem a filmem.  
**MLP** uczy się złożonych, nieliniowych wzorców.  
**NeuMF** łączy oba podejścia.

---

## Hiperparametry (domyślne)

| Parametr          | Wartość     |
|-------------------|-------------|
| Embedding size    | 32          |
| MLP layers        | [64, 32, 16]|
| Optimizer         | Adam        |
| Learning rate     | 0.001       |
| Batch size        | 256         |
| Negative samples  | 4           |
| Loss              | BCE         |
| Epoki             | 20          |

---

## Metryki i cele

| Metryka        | Cel (paper) | Opis                                   |
|----------------|-------------|----------------------------------------|
| HitRate@10     | > 0.65      | Czy trafny film jest w top-10          |
| NDCG@10        | > 0.38      | Jakość pozycji w rankingu top-10       |
| Poprawa vs pop | > 10%       | NeuMF vs popularność                   |

---

## Referencja

> Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua.  
> *Neural Collaborative Filtering.* WWW 2017.  
> https://arxiv.org/abs/1708.05031
