# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documento di riferimento

Il piano completo del progetto — issue azionabili, milestone, decisioni di design e principi non
negoziabili — è in **`docs/CLAUDE.md`**. È il documento autorevole: leggerlo prima di qualunque
modifica non banale, e seguirlo come ordine di esecuzione. Le sezioni qui sotto sono una sintesi
operativa per orientarsi rapidamente; in caso di conflitto, vince `docs/CLAUDE.md`.

## Stato del repo

- **Zero codice, zero dati.** Si parte dall'Issue #0 (scaffolding) della milestone M1.
- Branch principale: `main`.
- Struttura, `requirements.txt`, `config.yaml`, loader di config NON esistono ancora — vanno creati
  nell'ordine descritto in `docs/CLAUDE.md` (sezioni 2 e Issue #0).

## Perimetro e stack

- **Perimetro:** SOLO fase a eliminazione diretta del Mondiale 2026. Niente fase a gironi.
- **Stack:** Python 3.11+, `numpy`, `scipy.optimize`, `pandas`, `scikit-learn` (isotonic/Platt,
  metriche), `matplotlib`, `pyyaml`, `pytest`.
- **Niente framework di deep learning.** Si usa ottimizzazione convessa classica
  (Dixon-Coles via L-BFGS-B). Vedi Principio #1 in `docs/CLAUDE.md`.

## Lingua

- **Commenti e documentazione: italiano.**
- **Identificatori di codice (variabili, funzioni, classi, moduli): inglese.**

## Comandi previsti (una volta scaffoldato)

Questi comandi saranno disponibili dopo l'Issue #0; oggi falliscono perché i file non esistono.

```bash
pip install -r requirements.txt   # installa lo stack
pytest                            # esegue i test (anche solo smoke a inizio)
pytest tests/path/test_x.py::test_name   # esegue un singolo test
```

Non eseguire scaffolding "ad hoc" diverso da quello descritto: la struttura cartelle in
`docs/CLAUDE.md` sezione 2 è vincolante.

## Principi non negoziabili (sintesi)

I principi completi sono nella sezione 0 di `docs/CLAUDE.md`. **Non reintrodurre questi errori**:

1. **No reti neurali / deep learning.** Pochi dati, alta varianza: i modelli semplici vincono qui.
2. **Si modellano i gol** (Poisson bivariato + correzione Dixon-Coles), non l'esito binario
   "chi passa". Tutto il resto (90', supplementari, rigori, P(passa)) si deriva.
3. **Validazione SEMPRE temporale.** Mai split casuale: data leakage garantito.
4. **L'affidabilità si misura con la CALIBRAZIONE** (log-loss, Brier, curva di calibrazione),
   non con l'esito di un torneo né col profitto di poche scommesse.
5. **Il Monte Carlo PROPAGA, non crea affidabilità.** Aumentare le simulazioni riduce solo il
   rumore di campionamento.
6. **Edge di scommessa = disaccordo INFORMATO col mercato.** Esiste solo se `p` è calibrata.
7. **Si scommette presto (linea morbida).** Lo scoreboard onesto è il CLV, non vinto/perso.
8. **Inquadramento didattico:** Kelly frazionato, soglia minima di edge, cifre che si è sereni
   di perdere.

## Architettura (alta quota)

Pipeline lineare a milestone (M1 → M5; M6 è backlog avanzato):

```
data/raw            (Issue #1)      dataset Kaggle "International football results from 1872"
   ↓
data/processed      (Issue #2)      nomi normalizzati, neutral pulito, parquet
   ↓
src/data/weights    (Issue #3)      w = exp(-ξ·giorni) · w_competizione
   ↓
src/model/          (Issue #4-#8)   Dixon-Coles NLL → L-BFGS-B → α_i, β_i, γ, ρ
   ↓                                matrice 11×11 dei risultati → P(esiti 90')
   ↓                                cascata 90' / ET / rigori → P(passa)
   ↓
src/validation/     (Issue #9-#11)  split temporale, calibrazione isotonic, benchmark vs closing line
   ↓
src/market/         (Issue #12-#15) de-vig, blending modello+mercato (output A e B), edge, Kelly/4, CLV
   ↓
src/simulation/     (Issue #16)     Monte Carlo del bracket KO (default 50k iter, vettorizzato)
```

**Gate critico M3 → M4:** non si tocca il modulo scommesse finché la calibrazione del modello
non è dimostrata. Un edge calcolato su `p` scalibrate è privo di senso.

## Decisioni "fissate" vs "aperte"

- **Fissate** (sezione 1 di `docs/CLAUDE.md`): vanno nel `config.yaml`, **non vanno relitigate**.
  Includono perimetro, distribuzione gol, pesi competizione, supplementari/rigori, frazione Kelly,
  doppio output A/B.
- **Aperte** (da tunare per validazione, mai a occhio): ξ (decadimento temporale), w (blending col
  mercato), soglia minima di edge. La scelta avviene per grid search su log-loss out-of-sample.
- **Rimandate a M6:** xG, feature esogene (forma/infortuni/riposo), ensemble (Elo, XGBoost),
  valore rosa (Transfermarkt) come prior di shrinkage.

## Output che devono essere riproducibili da script

- **Curva di calibrazione** pre/post (Issue #10) — è il deliverable-prova del progetto.
- **Tabella metriche** modello vs baseline vs closing line (Issue #11).
- **Probabilità di torneo** per squadra (Issue #16).
- **Registro scommesse + CLV aggregato** (Issue #15).

Se uno di questi diventa "ottenibile solo a mano nel notebook", è un regresso.
