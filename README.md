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

## Modello Dixon-Coles (NLL pesata)

Cuore matematico del progetto, modulo `src/model/dixon_coles.py`.

**Modello dei gol** per una partita home `h` vs away `a`:

```
λ = exp(α[h] + β[a] + γ·is_home)   # gol attesi della casa
μ = exp(α[a] + β[h])                # gol attesi dell'ospite
```

con `is_home = 1` se la partita NON è in campo neutro, 0 altrimenti.

**Correzione Dixon-Coles** τ (raddrizza i 4 punteggi bassi correlati):

```
τ(0,0) = 1 − λ·μ·ρ
τ(0,1) = 1 + λ·ρ
τ(1,0) = 1 + μ·ρ
τ(1,1) = 1 − ρ
τ(x,y) = 1                          altrimenti

P(x,y) = τ(x,y; λ, μ, ρ) · Poisson(x; λ) · Poisson(y; μ)
```

**NLL pesata + penalty di identificabilità** (per rimuovere l'invarianza α→α+c, β→β−c):

```
NLL(θ) = −Σ_i w_i · log P(x_i, y_i; θ)
       + λ_id · ( mean(α)^2 + mean(β)^2 )
```

`λ_id` = `dixon_coles.identifiability_penalty_strength` (default `1e4`). Un floor su τ
(`tau_floor=1e-10`) evita `log(≤0)` quando il solver esplora la regione patologica.

Uso tipico:

```python
from src.model.indexing import TeamIndexer, prepare_match_data
from src.model.dixon_coles import initial_params, dixon_coles_nll_from_config
from src.config import load_config
import pandas as pd

df = pd.read_parquet("data/processed/matches_weighted.parquet")
indexer = TeamIndexer.from_match_dataframe(df)
data = prepare_match_data(df, indexer)
weights = df["weight"].to_numpy()
config = load_config()

p0 = initial_params(indexer.n_teams)                 # 2·315 + 2 = 632 parametri
nll = dixon_coles_nll_from_config(p0, data, weights, config)
```

## Fit del modello (L-BFGS-B)

```bash
python -m src.model.fit                            # default: ~minuti, salva data/models/dixon_coles.json
python -m src.model.fit --top 20                   # stampa le top 20 squadre per strength
python -m src.model.fit --max-fun 1000000          # budget alto se non converge
```

Minimizza la NLL Dixon-Coles con `scipy.optimize.minimize(method="L-BFGS-B")`, gradient
**numerico** (finite-diff). Bounds: `γ > 0`, `ρ ∈ [-0.2, 0.2]`, `α/β` liberi. Dopo il
fit, applica `center_alpha_beta` per ottenere `mean(α) = 0` (trasformazione invariante
del modello).

L'output `DixonColesModel` (JSON serializzato) contiene per ogni squadra `α` (attacco
sopra media in scala log-gol) e `β` (difesa: positivo = vulnerabile), più `γ` (vantaggio
casa), `ρ` (correzione DC), metadata di fit (NLL finale, n_iter, converged).

```python
from src.model.fit import DixonColesModel
model = DixonColesModel.load("data/models/dixon_coles.json")
model.to_dataframe().head(20)            # rating ordinati per α − β
```

**Limiti noti del modello base** (mitigabili in #17 con valore rosa Transfermarkt):
- bias confederazione: CONMEBOL gioca un campionato "tutti contro tutti" che espone
  forze relative più di altri continenti → rating sudamericani spesso gonfiati;
- nessun prior/shrinkage: squadre con poche partite hanno α/β molto rumorosi;
- niente xG, forma recente, infortuni — è il piano di M6.

## Esiti dei 90 minuti

Dato il modello fittato, `src/model/outcomes.py` calcola la matrice 11×11 dei punteggi
e i tre esiti aggregati (`P(home_win)`, `P(draw)`, `P(away_win)`) per una partita:

```python
from src.model.fit import DixonColesModel
from src.model.outcomes import match_outcome_90

model = DixonColesModel.load("data/models/dixon_coles.json")
out = match_outcome_90(model, "Italy", "Norway", is_neutral=True)
# Outcome90(p_home_win=0.376, p_draw=0.263, p_away_win=0.361,
#           expected_home_goals=1.45, expected_away_goals=1.42, ...)
```

La matrice `P(x, y)` viene costruita come outer product di due Poisson (λ, μ),
corretta sui 4 punteggi bassi con la Dixon-Coles `τ`, e rinormalizzata. La coda
fuori da 11×11 è ≪ 10⁻⁴ per partite con `λ, μ ≤ 2`; per partite molto sbilanciate
(top vs weak, λ alto), la rinormalizzazione conserva i 3 esiti aggregati ma altera
P(score=x) per x grandi (mai usato nella cascata).

`Outcome90.expected_home_goals = λ` e `expected_away_goals = μ` sono i tassi puri
che #7 riusa per i supplementari (`λ_ET = λ/3`, `μ_ET = μ/3`).

## Cascata 90'/supplementari/rigori → P(passa il turno)

Per una partita a eliminazione diretta, `src/model/cascade.py` compone i tre rami:

```
P(home passa) = P(home vince 90')
              + P(pari 90') · [ P(home vince ET) + P(pari ET) · P(home vince rigori) ]
```

```python
from src.config import load_config
from src.model.cascade import advance_probability_from_config
from src.model.fit import DixonColesModel

model = DixonColesModel.load("data/models/dixon_coles.json")
cfg = load_config()
out = advance_probability_from_config(model, "Spain", "England", cfg, is_neutral=True)
# AdvanceOutcome(
#   p_home_advance=0.563, p_away_advance=0.437,
#   p_home_win_90=0.396, p_draw_90=0.317, p_away_win_90=0.287,
#   p_home_win_et=0.241, p_draw_et=0.336, p_away_win_et=0.423,
#   p_home_win_penalty=0.500, penalty_favorite="home",
#   ...
# )
```

Tutti i passaggi della cascata sono esposti in `AdvanceOutcome` per debug e per il MC
del bracket (#16). I supplementari usano una mini-matrice con `λ_ET = λ · k_λ` e
`μ_ET = μ · k_μ` (default `k_λ = k_μ = 1/3`, configurabile in `config.yaml`); i rigori
sono una coin flip pesata con `+edge_to_favorite` al favorito sui 90' (default 0).

## Host policy Mondiale 2026 (USA/Canada/Mexico)

`src/model/host_policy.py` implementa la policy del vantaggio casa per il Mondiale
tri-ospitante. **Assunzione esplicita** (nessun precedente storico):

- γ **pieno** se la squadra gioca nel proprio paese ospitante (es. USA in USA);
- γ **ridotto a metà** se un host gioca in un altro dei tre paesi ospitanti
  (es. USA a Toronto);
- γ = 0 (campo neutro effettivo) in tutti gli altri casi (es. Argentina in USA, o
  USA in Italia).

Configurabile in `config.yaml → host_advantage_2026`. La policy si attiva passando
`venue_country` alle API esistenti:

```python
from src.model.cascade import advance_probability_from_config

# USA in casa propria vs Argentina
out = advance_probability_from_config(model, "United States", "Argentina", cfg,
                                       venue_country="United States")
# out.is_neutral == False, out.p_home_advance riflette γ pieno

# Argentina in USA (venue host ma squadra non-host) → effettivamente neutro
out = advance_probability_from_config(model, "Argentina", "Brazil", cfg,
                                       venue_country="United States")
# out.is_neutral == True (scale di γ = 0)
```

Priorità dei parametri (top→bottom): `home_advantage_scale` esplicito > `venue_country` +
`host_policy` > `is_neutral`.

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
