# Jak to działa — wyjaśnienie dla każdego

Bez żargonu. Od początku.

---

## Problem: skąd Netflix wie, co chcesz oglądać?

Masz 6000 użytkowników i 4000 filmów. Każdy użytkownik obejrzał i ocenił może 50–100 filmów. Reszta — nieznana. Chcemy przewidzieć, które **nieobejrzane** filmy dany użytkownik polubi.

To jest dokładnie ten problem. Dataset MovieLens-1M to milion ocen od prawdziwych ludzi — używamy go, żeby nauczyć model "myśleć" o gustach filmowych.

---

## Krok 1 — Dane (`data/dataset.py`)

### Skąd bierzemy dane
Plik `dataset.py` automatycznie pobiera dataset MovieLens-1M z internetu przy pierwszym uruchomieniu. Dostajemy tabelę: `użytkownik → film → ocena (1-5)`.

### Uproszczenie: lubię / nie wiem
Zamiast przewidywać dokładną ocenę (trudne), upraszczamy:
- ocena ≥ 4 → **"lubię"** (1)
- ocena < 4 lub brak oceny → **"nie wiem"** (0)

To się nazywa *implicit feedback* — zamiast "jak bardzo lubisz" pytamy tylko "czy lubisz".

### Podział danych
Dla każdego użytkownika bierzemy jego oceny posortowane chronologicznie i dzielimy:
- najstarsze 80% → **trening** (model się uczy)
- kolejne 10% → **walidacja** (sprawdzamy czy model nie jest za mądry na dane treningowe)
- najnowsze 10% → **test** (finalna ocena)

Dlaczego chronologicznie? Bo to uczciwe — w realu też nie mamy dostępu do przyszłości.

### Fałszywe negatywy (negative sampling)
Model musi się uczyć nie tylko tego co lubisz, ale też czego "nie lubisz". Problem: brak oceny ≠ nielubienie — może po prostu nie widziałeś filmu. Dlatego losujemy filmy, z którymi użytkownik **nigdy** nie wchodził w interakcję i traktujemy je jako negatywy. Na każdy 1 pozytyw losujemy 4 negatywy.

---

## Krok 2 — Model (`models/ncf.py`)

### Czym jest embedding?

Każdy użytkownik i każdy film dostaje swój **wektor liczb** (np. 32 liczby). To jest *embedding*. Możesz to sobie wyobrazić jako "odcisk palca" użytkownika albo "DNA" filmu — zestaw liczb, który opisuje jego charakterystykę.

Na początku są losowe. Model uczy się je zmieniać tak, żeby podobni użytkownicy mieli podobne wektory, a filmy, które dany użytkownik lubi — też były "blisko" jego wektora.

### GMF — prosta wersja

Bierzemy wektor użytkownika i wektor filmu, mnożymy je element po elemencie i sumujemy. To daje jedną liczbę — im wyższa, tym bardziej pasują do siebie.

Problem: to jest operacja liniowa. Nie jest w stanie uchwycić skomplikowanych zależności — np. "lubię filmy akcji, ale tylko te z lat 90.".

### MLP — sprytna wersja

Zamiast mnożenia, sklejamy oba wektory obok siebie i przepuszczamy przez kilka warstw sieci neuronowej. Każda warstwa może się nauczyć dowolnie skomplikowanego wzorca. To jak mózg, który widzi parę (użytkownik, film) i może wykryć subtelne powiązania.

### NeuMF — połączenie obu

Skoro każde podejście widzi coś innego, dlaczego nie użyć obu naraz? GMF i MLP pracują równolegle na tych samych danych, a ich wyniki są łączone i przekazywane do końcowej warstwy, która mówi: "tak, ten użytkownik polubi ten film" albo "nie".

```
Użytkownik + Film
       │
  ┌────┴────┐
  GMF      MLP
  │         │
  └────┬────┘
    łączenie
       │
    wynik (0–1)
```

---

## Krok 3 — Trening (`train.py`)

Model startuje z losowymi embeddingami i powoli je poprawia. Dla każdej pary (użytkownik, film) z labelem 1 lub 0:

1. Model przewiduje liczbę z zakresu 0–1
2. Porównujemy z prawdziwą odpowiedzią (BCE loss — kara za złe przewidywanie)
3. Model lekko poprawia swoje wektory żeby następnym razem być bliżej prawdy

Robimy to milion razy (cały dataset × 20 epok). Optimizer Adam dba o to żeby kroki były rozsądnej wielkości.

**Early stopping** — jeśli model przestaje się poprawiać na danych walidacyjnych przez 5 epok z rzędu, zatrzymujemy trening wcześniej (żeby nie przegiąć).

---

## Krok 4 — Ewaluacja (`evaluate.py`)

Jak sprawdzić czy model jest dobry? Nie patrzymy na dokładność przewidywania ocen, tylko na **jakość rankingu**.

Dla każdego użytkownika:
1. Model ocenia wszystkie filmy, których użytkownik nie widział
2. Bierzemy top 10 najwyżej ocenionych
3. Sprawdzamy czy film, który użytkownik naprawdę lubił (z zbioru testowego), znalazł się w tych 10

**HitRate@10** — ile procent użytkowników miało swój "trafiony" film w top 10  
**NDCG@10** — to samo, ale z nagrodą za wyższe pozycje (miejsce 1 jest warte więcej niż miejsce 10)

Wartości z oryginalnego papieru naukowego: HitRate > 0.65 i NDCG > 0.38. Dążymy do tych liczb.

---

## Krok 5 — Eksperymenty (`notebooks/`)

### ablation.ipynb
Sprawdzamy co wnosi każda część modelu:
- Jak dobrze działa sam GMF?
- Jak dobrze działa sam MLP?
- Czy NeuMF (oba razem) jest lepszy od każdego z osobna?
- Czy większe embeddingi (więcej liczb w wektorze) dają lepsze wyniki?

### results.ipynb
Końcowe wyniki: wykresy, porównanie z najprostszym możliwym baseline'm (po prostu rekomenduj najpopularniejsze filmy wszystkim), i przykładowe rekomendacje — co model proponuje konkretnym użytkownikom z bazy.

---

## Uproszczone podsumowanie

| Co                  | Jak                                                |
|---------------------|----------------------------------------------------|
| Dane                | 1M ocen filmów, uproszczone do "lubię / nie wiem"  |
| Model               | Sieć neuronowa ucząca się gustów z historii ocen   |
| Trening             | Pokazujemy pary (użytkownik, film) i uczymy model  |
| Ocena jakości       | Czy trafny film jest w top 10 rekomendacji?        |
| Cel                 | Lepiej niż "polecaj każdemu to co popularne"       |
