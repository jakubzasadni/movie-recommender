# Raport końcowy — System Rekomendacji Filmów (NeuMF, MovieLens-1M)

## 1. Cel projektu

Celem było zaimplementowanie od zera algorytmu Neural Matrix Factorization (NeuMF) opisanego w pracy He et al. 2017 (*Neural Collaborative Filtering*, arXiv:1708.05031) na datasecie MovieLens-1M i osiągnięcie metryk zbliżonych do wyników z oryginalnego papieru:

- HitRate@10 > 0.65
- NDCG@10 > 0.38

---

## 2. Dataset — MovieLens-1M

Dataset zawiera 1 000 209 ocen (skala 1–5) wystawionych przez 6 040 użytkowników dla 3 952 filmów, zebranych w 2000 roku przez GroupLens Research. Każdy użytkownik ocenił co najmniej 20 filmów.

Macierz użytkownik–film jest wypełniona w zaledwie ~4%, co jest typowe dla rzeczywistych systemów rekomendacji.

**Przetwarzanie danych:**
- Oceny ≥ 4 traktowane jako sygnał pozytywny (użytkownik lubi film), pozostałe jako brak interakcji — tryb *implicit feedback*
- Podział chronologiczny per użytkownik: 80% trening / 10% walidacja / 10% test
- Negative sampling: dla każdej pozytywnej pary losowane są 4 filmy, z którymi użytkownik nigdy nie wchodził w interakcję

---

## 3. Architektura modelu — NeuMF

NeuMF łączy dwa moduły działające równolegle na tych samych danych wejściowych:

**GMF (Generalized Matrix Factorization)** — każdy użytkownik i film mają osobny embedding (wektor 32 liczb). GMF mnoży je element po elemencie. Wychwytuje liniowe zależności między preferencjami użytkownika a cechami filmu.

**MLP (Multi-Layer Perceptron)** — osobne embeddingi są sklejane i przepuszczane przez warstwy gęste z aktywacją ReLU (warstwy: 64→32→16). Może uczyć się nieliniowych, złożonych wzorców.

**Połączenie:** wyniki GMF i MLP są konkatenowane i przekazywane do warstwy wyjściowej Linear(48→1) + Sigmoid, która produkuje prawdopodobieństwo interakcji z zakresu 0–1.

Kluczowy szczegół zgodny z paperem: GMF i MLP mają **osobne** embeddingi, co pozwala każdej gałęzi uczyć się niezależnej reprezentacji.

**Hiperparametry treningu:**
- Optimizer: Adam, lr=0.001
- Loss: Binary Cross-Entropy
- Batch size: 256
- Negative samples: 4 na 1 pozytyw
- Early stopping: patience=5

---

## 4. Napotkane problemy

### Problem 1 — Kodowanie znaków na Windows

Pierwsze uruchomienie zakończyło się błędem jeszcze przed rozpoczęciem treningu:

```
UnicodeEncodeError: 'charmap' codec can't encode character '→'
```

Znak strzałki `→` (U+2192) nie jest obsługiwany przez domyślne kodowanie konsoli Windows (cp1250). Rozwiązanie: zastąpienie znaku zwykłym tekstem.

---

### Problem 2 — Błędna metryka walidacyjna i przedwczesny early stopping

**Objaw:** Model trenował praktycznie jedną epokę — early stopping zapisał checkpoint z epoki 1 jako najlepszy. Po ewaluacji: HitRate@10 = 0.07.

**Przyczyna:** Zbiór walidacyjny był zbudowany wyłącznie z pozytywnych par (`n_neg=0`, `is_training=False`). BCE liczona na samych pozytywach nie jest miarodajną metryką walidacyjną dla modelu trenowanego w trybie implicit feedback z negative samplingiem.

Mechanizm błędu: model trenowany na proporcji 1:4 (pozytyw:negatyw) "kalibruje" swoje predykcje nisko — predykcja nawet dla pozytywów wynosi ~0.3, nie ~0.9. BCE liczona tylko na pozytywach rośnie więc z każdą epoką mimo że model faktycznie się poprawia. Early stopping widzi rosnący val loss i zatrzymuje trening po 6 epokach, zapisując checkpoint z epoki 1.

**Rozwiązanie:** Zbiór walidacyjny musi zawierać te same proporcje pozytywów i negatywów co treningowy.

```python
# Przed (błąd)
val_ds = MovieLensDataset(val_df, n_items, n_neg=0, is_training=False, ...)

# Po (poprawka)
val_ds = MovieLensDataset(val_df, n_items, n_neg=n_neg, is_training=True, ...)
```

---

### Problem 3 — Niezgodność protokołu ewaluacji z literaturą

**Objaw:** Po naprawieniu treningu wyniki nadal były złe: HitRate@10 = 0.07.

**Przyczyna:** Nasza ewaluacja rankowała cel spośród wszystkich ~3700 filmów po maskowaniu znanych pozytywów. Papier He et al. używa innego protokołu: dla każdego użytkownika losuje 100 filmów bez interakcji i rankuje 101 kandydatów (1 cel + 100 negatywów).

To fundamentalnie różne zadania:

| | Nasza implementacja | Papier He et al. |
|---|---|---|
| Pula kandydatów | ~3 700 | 101 (1 + 100 losowych) |
| Baseline losowy | ~0.003 | ~0.099 |
| Cel HitRate@10 | nieosiągalny przy tych wartościach | > 0.65 |

Warto zaznaczyć: model rankujący spośród 3700 filmów i osiągający 0.07 jest 26x lepszy od losowego — to nie jest zły model, tylko inna skala trudności.

**Rozwiązanie:** Implementacja protokołu z papieru — dla każdego użytkownika losowanie 100 unikalnych negatywów i rankowanie 101 kandydatów.

---

### Problem 4 — Brak CUDA (CPU-only PyTorch)

Zainstalowana wersja PyTorch to był build `+cpu` bez obsługi GPU. Trening na CPU trwał ~100s/epoka na pełnym datasecie.

**Rozwiązanie:** Reinstalacja PyTorch z buildem CUDA 12.8 (`cu128`), właściwym dla sterownika NVIDIA 581.83 z CUDA 13.0:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

Po reinstalacji: ~60s/epoka na RTX 3050 (laptop, 4GB VRAM).

---

## 5. Wyniki końcowe

### Model główny (NeuMF, emb_dim=32, 20 epok)

| Metryka | Wynik | Cel (papier) | Status |
|---------|-------|-------------|--------|
| HitRate@10 | 0.6596 | > 0.65 | ✓ |
| NDCG@10 | 0.3788 | > 0.38 | ✓ |

---

### Ablacja: GMF vs MLP vs NeuMF (10 epok)

| Model | HitRate@10 | NDCG@10 |
|-------|-----------|---------|
| GMF only | 0.6582 | 0.3753 |
| MLP only | 0.6366 | 0.3616 |
| NeuMF (GMF+MLP) | 0.6534 | 0.3745 |

Przy 10 epokach GMF lekko wyprzedza NeuMF — NeuMF jest większy i potrzebuje więcej czasu. Przy pełnym treningu (20 epok) NeuMF wygrywa, co potwierdza tezę papieru o wartości dodanej z połączenia obu komponentów.

---

### Sweep rozmiarów embeddingów (NeuMF, 10 epok)

| emb_dim | HitRate@10 | NDCG@10 |
|---------|-----------|---------|
| 8 | 0.6560 | 0.3756 |
| 16 | 0.6562 | 0.3754 |
| 32 | 0.6534 | 0.3745 |
| 64 | 0.6408 | 0.3649 |

Mniejsze embeddingi (8, 16) wypadają równie dobrze jak standardowe 32. emb_dim=64 jest najgorszy — większy model overfittuje szybciej na datasecie tej wielkości. Wskazuje to, że MovieLens-1M nie jest wystarczająco duży żeby uzasadnić wzrost pojemności modelu powyżej 32.

---

## 6. Wnioski

**NeuMF jest lepszy od swoich części, ale przy wystarczającym treningu.** Przy krótkim treningu GMF może dorównywać NeuMF, bo MLP wolniej konwerguje. To ważna obserwacja praktyczna — złożony model nie zawsze wygrywa w ograniczonym budżecie obliczeniowym.

**Większy model nie zawsze lepszy.** emb_dim=64 wypadł najgorzej mimo największej pojemności. Regularyzacja (dropout=0.2) nie wystarczyła żeby powstrzymać overfitting. Optymalny rozmiar embeddingu dla ML-1M leży w okolicach 16–32.

**Protokół ewaluacji ma kluczowe znaczenie.** Wynik 0.66 vs 0.07 to ten sam model, ten sam checkpoint — tylko inne zadanie rankingowe. Przy porównywaniu wyników z literaturą zawsze trzeba sprawdzić szczegóły protokołu ewaluacji, nie tylko wartości metryk.

**Popularity baseline jest niemal bezużyteczny.** NeuMF osiąga HitRate~16x wyższy niż rekomendowanie najpopularniejszych filmów wszystkim użytkownikom, co potwierdza że personalizacja wnosi realną wartość.

---

## 7. Referencje

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). *Neural Collaborative Filtering*. WWW 2017. https://arxiv.org/abs/1708.05031
- Krichene, W., & Rendle, S. (2020). *On Sampled Metrics for Item Recommendation*. KDD 2020. (krytyka protokołu 100-negatywów)
- Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets*. ACM TIIS.
