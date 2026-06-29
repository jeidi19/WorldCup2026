# WorldCup2026

Modello probabilistico per la **fase a eliminazione diretta** del Mondiale 2026 + modulo
scommesse a fini **didattici/sperimentali**.

Si modellano i **gol** (Poisson bivariato con correzione Dixon-Coles); da lì si derivano
gli esiti dei 90', dei supplementari, dei rigori e la probabilità di passare il turno.
Niente reti neurali, validazione **sempre** temporale, l'affidabilità si misura con la
**calibrazione** (log-loss, Brier, curva di calibrazione) non con l'esito di un torneo.

Il piano completo (principi non negoziabili, decisioni fissate vs aperte, 19 issue
azionabili su 6 milestone) è in **`docs/CLAUDE.md`**.

## Setup

Requisiti: **Python 3.11+**.

```bash
git clone https://github.com/jeidi19/WorldCup2026.git
cd WorldCup2026

# Crea e attiva un virtualenv
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

# Installa le dipendenze
pip install -r requirements.txt

# Verifica setup (config caricabile + smoke test)
pytest -q
```

## Acquisizione dati

Il dataset storico delle partite tra nazionali è scaricato da Kaggle
([`martj42/international-football-results-from-1872-to-2017`](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017),
aggiornato al presente nonostante il nome) tramite `kagglehub`.

**Setup credenziali.** Crea un file `.env` alla radice del repo (è in `.gitignore`):

```bash
# .env
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Genera il token su [kaggle.com/settings](https://www.kaggle.com/settings) → *API* → *Create New Token*.

**Esecuzione del download:**

```bash
python -m src.data.download
```

Output:
- `data/raw/results.csv` — dataset grezzo (immutabile, mai modificato in-place).
- `data/raw/metadata.json` — fonte, data download, hash SHA-256, righe, colonne, range temporale.

## Pulizia dati

```bash
python -m src.data.clean
```

Genera `data/processed/matches_clean.parquet` applicando:
- Drop righe con NaN sui punteggi (tipicamente partite future già schedulate).
- Drop entità non-FIFA / regionali / micronazioni (deny-list versionata in
  `src/data/team_policies.py`).
- Normalizzazione nomi (alias documentati, attualmente vuoto).
- `neutral` coerced a booleano puro.
- `match_id` come hash SHA-1 troncato di `(date|home|away)` — stabile tra rebuild.
- Ordine cronologico crescente.

**Policy sulle entità storiche.** Le squadre estinte (German DR, Yugoslavia, Czechoslovakia,
North Vietnam, Vietnam Republic) sono MANTENUTE come segnale (servono per stimare le forze
delle squadre superstiti), ma non saranno predette al Mondiale 2026 perché non sono in
tabellone. Vedi `src/data/team_policies.py` per i dettagli.

## Pesatura partite (per il fit Dixon-Coles)

```bash
python -m src.data.build_weights
# oppure con reference_date custom (per la validazione temporale)
python -m src.data.build_weights --reference-date 2022-12-31
```

Genera `data/processed/matches_weighted.parquet` (tutte le colonne + `weight`) e un
`weights_metadata.json` con: reference_date usata, emivita, statistiche dei pesi, lista dei
tornei non mappati.

**Peso finale:** `w = w_time · w_comp`.

- **w_time** = `exp(-ξ · giorni_fa)`, con `ξ = ln(2) / (emivita · 365.25)`. Emivita di default
  2 anni (vedi `time_decay.half_life_years` in `config.yaml`, grid tunabile in #9). Una partita
  con `date == reference_date` pesa 1; a una emivita pesa 0.5.
- **w_comp** = moltiplicatore dal bucket del torneo (vedi `src/data/competition_policies.py`):
  - 1.0 — `mondiali` (FIFA World Cup, Confederations Cup, Finalissima)
  - 1.0 — `finali_continentali` (Euro, Copa América, AFCON, Asian Cup, Gold, OFC)
  - 0.8 — `qualificazioni` (tutti i `... qualification`, anche fallback pattern)
  - 0.8 — `nations_league` (UEFA / CONCACAF Nations League)
  - 0.6 — `sub_continentali` (CECAFA, COSAFA, Gulf Cup, AFF, SAFF, EAFF, WAFF, UNCAF, ...)
  - 0.4 — `amichevole` (Friendly, Merdeka, BHC, Nordic, King's Cup, Korea Cup, ...)
  - 0.6 — `default_unmapped` per tornei non riconosciuti (loggati per revisione)

**Drop di pre-fit:** i tornei **multi-sport / U-23** (Asian Games, SEA Games, Island Games,
Pacific Games, Pan American Games, ...) vengono rimossi prima del calcolo dei pesi: schierano
formazioni B / U-23 e contaminerebbero la stima delle forze delle nazionali A.

## Struttura

```
WorldCup2026/
├── config.yaml                 # parametri fissati e aperti (vedi docs/CLAUDE.md §1)
├── requirements.txt
├── pyproject.toml              # config pytest
├── data/
│   ├── raw/                    # dataset originale (NON committato, scaricato a runtime)
│   └── processed/              # output puliti (NON committati)
├── src/
│   ├── config.py               # loader Pydantic v2 della configurazione
│   ├── data/                   # acquisizione + pulizia + pesi (Issue #1–#3)
│   ├── model/                  # Dixon-Coles, fit, matrice, cascata (Issue #4–#8)
│   ├── validation/             # split temporale, calibrazione, metriche (Issue #9–#11)
│   ├── market/                 # de-vig, blending, edge, Kelly, CLV (Issue #12–#15)
│   └── simulation/             # Monte Carlo del bracket (Issue #16)
├── notebooks/                  # esplorazione + grafici
├── tests/
└── docs/
    └── CLAUDE.md               # piano completo del progetto (autorevole)
```

## Configurazione

Tutti i parametri stanno in **`config.yaml`** e sono caricabili tipizzati:

```python
from src.config import load_config

config = load_config()
print(config.kelly.fraction)               # 0.25
print(config.time_decay.half_life_years)   # 2.0
```

- I parametri **"fissati"** (perimetro, pesi competizione, Kelly frazionato, ecc.)
  non vanno relitigati: sono decisioni di design (vedi `docs/CLAUDE.md` §0–§1).
- I parametri **"aperti"** (`xi` di decadimento temporale, `w` di blending col mercato,
  soglia minima di edge) vanno scelti per **validazione**, mai a mano.

## Lingua

- **Commenti e doc:** italiano.
- **Identificatori di codice:** inglese.

## Stato

In sviluppo, milestone M1 (fondamenta dati). Per il backlog completo:
[milestone su GitHub](https://github.com/jeidi19/WorldCup2026/milestones).

## Inquadramento didattico

Il modulo scommesse, quando arriverà (M4), serve a misurare il **CLV** (Closing Line Value)
e a illustrare il rapporto edge/calibrazione/Kelly frazionato. Puntare solo cifre che si è
sereni di perdere; nessuna garanzia di profitto.
