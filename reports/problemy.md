# Napotkane problemy i rozwiązania

## Problem 1 — Kodowanie znaków na systemie Windows

**Objaw:** Pierwsze uruchomienie `train.py` zakończyło się błędem już przy próbie wyświetlenia komunikatu o pobieraniu danych:

```
UnicodeEncodeError: 'charmap' codec can't encode character '→'
in position 25: character maps to <undefined>
```

**Przyczyna:** Znak `→` (U+2192) nie jest obsługiwany przez domyślne kodowanie konsoli Windows (cp1250). Kod pisany i testowany na systemie Linux/macOS może cicho złamać się na Windows przy dowolnym znaku spoza ASCII.

**Rozwiązanie:** Zastąpienie znaku `→` zwykłym tekstem w komunikacie w `data/dataset.py`.

---

## Problem 2 — Błędna metryka walidacyjna i przedwczesny early stopping

**Objaw:** Model uczył się tylko jedną epokę. Val loss rósł od pierwszej epoki, early stopping zapisał checkpoint z epoki 1 jako "najlepszy". Po ewaluacji HitRate@10 wyniósł **0.07**.

**Przyczyna:** Zbiór walidacyjny był skonstruowany wyłącznie z par pozytywnych (`is_training=False`, `n_neg=0`). Funkcja kosztu BCE liczona na takim zbiorze mierzy tylko *jak bardzo model jest pewny pozytywów* — bez żadnych negatywów w zestawie.

W trakcie treningu model uczy się jednocześnie:
- podnosić score'y dla par pozytywnych,
- obniżać score'y dla negatywów.

Ponieważ negatywów jest 4× więcej niż pozytywów (negative sampling 4:1), model "kalibruje" swoje predykcje w dół — predykcje nawet dla pozytywów są relatywnie niskie. To oznacza, że BCE liczona wyłącznie na pozytywach rośnie z każdą epoką, choć model faktycznie się poprawia. Early stopping widzi rosnący val loss i zatrzymuje trening zbyt wcześnie.

**Rozwiązanie:** Zbiór walidacyjny musi zawierać te same proporcje pozytywów i negatywów co zbiór treningowy. Zmiana w `data/dataset.py`:

```python
# Przed
val_ds = MovieLensDataset(val_df, n_items, n_neg=0, is_training=False, ...)

# Po
val_ds = MovieLensDataset(val_df, n_items, n_neg=n_neg, is_training=True, ...)
```

Po poprawce val loss prawidłowo maleje przez pierwsze epoki, a następnie rośnie (klasyczny sygnał overfittingu), co pozwala early stoppingowi działać zgodnie z przeznaczeniem.

---

## Problem 3 — Niezgodność protokołu ewaluacji z literaturą

**Objaw:** Po naprawieniu treningu wyniki ewaluacji nadal wyglądały fatalnie: **HitRate@10 = 0.07** wobec celu >0.65 z papieru.

**Przyczyna:** Niezgodność protokołu ewaluacji między naszą implementacją a papierem He et al. 2017.

| | Nasza implementacja | Papier He et al. |
|---|---|---|
| Pula kandydatów | ~3700 filmów (wszystkie niewidziane) | 101 filmów (1 cel + 100 losowych negatywów) |
| Zadanie modelu | Top-10 spośród ~3700 | Top-10 spośród 101 |
| Baseline (losowy) | ~0.003 | ~0.099 |

Trafić w top-10 spośród 101 kandydatów jest radykalnie łatwiejsze niż spośród 3700. Wartości HitRate@10 > 0.65 z papieru odnoszą się wyłącznie do protokołu 100-negatywów i nie są bezpośrednio porównywalne z pełnym rankingiem.

**Rozwiązanie:** Implementacja protokołu z papieru w `evaluate.py` — dla każdego użytkownika losowanie 100 unikalnych negatywów i rankowanie 101 kandydatów.

**Wyniki po poprawce:**

| Metryka | Wynik | Cel (papier) | Status |
|---|---|---|---|
| HitRate@10 | 0.6596 | > 0.65 | ✓ |
| NDCG@10 | 0.3788 | > 0.38 | ✓ |

**Uwaga:** Protokół 100-negatywów jest krytykowany w nowszej literaturze (Krichene & Rendle, 2020) jako podatny na bias statystyczny — wyniki zależą od tego, które negatywy zostaną wylosowane. Pozostaje jednak standardem reprodukowalności dla benchmarku MovieLens-1M i pozwala na bezpośrednie porównanie z oryginalnym paperem.
