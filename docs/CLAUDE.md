# HANDOFF per Claude Code — Modello di predizione Mondiali 2026 (fase a eliminazione)

> **Cos'è questo documento.** È il piano di lavoro per implementare da zero un modello
> di predizione probabilistica delle partite a **eliminazione diretta** del Mondiale 2026,
> con un modulo scommesse a fini **didattici/sperimentali**. È diviso in **issue azionabili**,
> in ordine di dipendenza, ognuna con obiettivo, task, criteri di completamento (Definition of
> Done) e note tecniche. Lavora le issue nell'ordine dato salvo dipendenze esplicite.
>
> **Stato attuale:** zero codice, zero dati. Si parte dall'Issue #0.
> **Perimetro:** SOLO fase a eliminazione (no fase a gironi).
> **Lingua:** commenti e doc in italiano; identificatori di codice in inglese.

---

## 0. Principi non negoziabili (NON reintrodurre questi errori)

Questi punti sono stati decisi consapevolmente. Non vanno "ottimizzati via" né reintrodotti.

1. **NIENTE reti neurali / deep learning.** Pochi dati (migliaia di partite) + varianza enorme
   del calcio. Si usa ottimizzazione convessa classica (Dixon-Coles via L-BFGS-B). Modelli
   semplici battono il DL qui.
2. **Non si modella l'esito binario "chi passa".** Si modellano i **gol** (Poisson bivariato
   corretto Dixon-Coles) e da lì si deriva tutto: 90', supplementari, rigori, P(passa).
3. **Validazione SEMPRE temporale.** Allena sul passato, valida sul futuro. MAI split casuale
   (data leakage garantito).
4. **L'affidabilità si misura con la CALIBRAZIONE**, non con l'esito di un torneo né col profitto
   di poche scommesse. Metriche vere: log-loss, Brier, curva di calibrazione. Benchmark di
   riferimento: la **closing line** del mercato sulle stesse partite storiche.
5. **Il Monte Carlo PROPAGA, non crea affidabilità.** Aumentare le simulazioni riduce solo il
   rumore di campionamento, non avvicina alla realtà. Garbage in → garbage out, ma quantificato.
6. **L'edge di scommessa = disaccordo INFORMATO col mercato**, non prevedibilità. Esiste solo se
   la `p` del modello è calibrata. Edge su `p` scalibrata = perdita mascherata.
7. **Si scommette PRESTO (linea morbida), non alla chiusura.** "Battere la closing line" significa
   prendere quote migliori di quelle a cui la linea chiuderà, non aspettare la chiusura. Lo scoreboard
   onesto è il **CLV (Closing Line Value)**, non il vinto/perso.
8. **Inquadramento didattico.** Puntare solo cifre che si è sereni di perdere; soglia minima di edge;
   Kelly **frazionato**.

---

## 1. Decisioni fissate (config) vs aperte (da tunare)

### Fissate — mettere in `config.yaml`, NON relitigare
| Parametro | Valore di partenza | Note |
|---|---|---|
| Perimetro | solo eliminazione diretta | niente gironi |
| Distribuzione gol | Poisson + correzione Dixon-Coles (ρ) | ρ = parametro di dipendenza globale |
| Pesi competizione | amichevole 0.4 · qualificazioni/Nations 0.8 · finali continentali/Mondiali 1.0 | fissi a mano (evitare overfitting) |
| Includere amichevoli | **SÌ**, down-pesate | servono come ponti cross-confederazione; NON escluderle |
| Supplementari | λ_ET = λ/3, μ_ET = μ/3 | mini-matrice da 0-0 |
| Rigori | 0.50 (eventualmente 0.52 al favorito sui 90') | non è lì che si vince/perde |
| Kelly | frazionato a 1/4 | mai Kelly pieno |
| Output | DOPPIO: A "calibrato" (blend pesante vs mercato) + B "modello puro" | vedi Issue #13 |

### Aperte — da decidere PER VALIDAZIONE (grid search su log-loss out-of-sample)
- **ξ (decadimento temporale):** centro emivita ~2 anni; grid su emivita ∈ [1, 4] anni.
  Conversione: `ξ = ln(2) / emivita_giorni`.
- **w (peso del blending col mercato):** alto per l'output calibrato (A), concettualmente 0 per
  l'output puro (B). Scegliere w di A che minimizza log-loss sul futuro held-out.
- **Soglia minima di edge** per scommettere (partenza: 3–4%).

### Rimandate a milestone avanzata (NON in v1)
- **xG** (expected goals): scarsi e disomogenei per le nazionali → fragili come feature core.
- **Feature esogene** (forma, infortuni, riposo) e **ensemble** (Elo, XGBoost).
- **Valore rosa (Transfermarkt)** come prior per lo shrinkage (prima feature da aggiungere quando si parte con le esogene).

---

## 2. Stack e struttura repo

**Stack:** Python 3.11+, `numpy`, `scipy` (optimize), `pandas`, `scikit-learn` (isotonic/Platt,
metriche), `matplotlib` (curva di calibrazione), `pyyaml`, `pytest`. **Niente** framework DL.

```
WorldCup2026/
├── README.md
├── requirements.txt
├── config.yaml                 # iperparametri e pesi (sezione 1)
├── data/
│   ├── raw/                     # dataset scaricati, immutabili
│   └── processed/               # output puliti
├── src/
│   ├── data/                    # Issue #1–#3
│   ├── model/                   # Issue #4–#8
│   ├── validation/              # Issue #9–#11
│   ├── market/                  # Issue #12–#15
│   └── simulation/              # Issue #16
├── notebooks/                   # esplorazione + grafici
└── tests/
└── docs/
```

---

## 3. Milestone e ordine di esecuzione

- **M1 — Fondamenta dati:** Issue #0 → #3
- **M2 — Modello core:** Issue #4 → #8
- **M3 — Validazione & calibrazione:** Issue #9 → #11  *(qui si dimostra se il modello vale qualcosa)*
- **M4 — Mercato & scommesse:** Issue #12 → #15
- **M5 — Simulazione torneo:** Issue #16
- **M6 — Avanzato (backlog):** Issue #17 → #18

> **Gate critico:** non passare a M4 (scommesse) se M3 non mostra una calibrazione decente.
> Un edge calcolato su probabilità scalibrate è privo di senso (Principio #6).

---

## ISSUE #0 — Scaffolding del progetto
**Milestone:** M1 · **Dipende da:** nessuna

**Obiettivo:** struttura repo, dipendenze, config caricabile.

**Task:**
- Creare la struttura cartelle della sezione 2.
- `requirements.txt` con lo stack della sezione 2.
- `config.yaml` con tutti i parametri "fissati" e "aperti" della sezione 1, con valori di default.
- Loader del config (`src/config.py`) + `README.md` con istruzioni di setup.

**Definition of Done:**
- [ ] `pip install -r requirements.txt` va a buon fine.
- [ ] Il config si carica e i valori sono accessibili come oggetto tipizzato.
- [ ] `pytest` gira (anche solo con uno smoke test).

---

## ISSUE #1 — Acquisizione dati storici
**Milestone:** M1 · **Dipende da:** #0

**Obiettivo:** scaricare lo storico delle partite tra nazionali.

**Task:**
- Usare il dataset Kaggle **"International football results from 1872"** (`results.csv`:
  date, home_team, away_team, home_score, away_score, tournament, neutral).
- Salvare in `data/raw/` immutabile. Documentare fonte e data di download nel README.
- Script `src/data/download.py` (o istruzioni manuali se l'API Kaggle non è disponibile).

**Definition of Done:**
- [ ] CSV grezzo in `data/raw/`, mai modificato in-place.
- [ ] Conteggio righe e range temporale loggati.

**Note:** la colonna `neutral` (campo neutro sì/no) è essenziale per stimare il vantaggio casa (Issue #8).

---

## ISSUE #2 — Pulizia e normalizzazione
**Milestone:** M1 · **Dipende da:** #1

**Obiettivo:** dataset pulito e coerente.

**Task:**
- Normalizzare i nomi delle nazionali (gestire rinomine storiche: es. Germany Ovest/Est, Yugoslavia,
  ecc. — decidere una policy e documentarla; per il 2026 contano le entità attuali).
- Tipizzare le date, droppare/righe malformate, gestire NaN sui punteggi.
- Mantenere `neutral` come flag booleano pulito.
- Output in `data/processed/matches_clean.parquet`.

**Definition of Done:**
- [ ] Nessun nome squadra ambiguo/duplicato non risolto (lista mappature in un file versionato).
- [ ] Test: nessun NaN nei campi chiave; date monotone ordinabili.

---

## ISSUE #3 — Pesi (temporale + competizione)
**Milestone:** M1 · **Dipende da:** #2

**Obiettivo:** assegnare a ogni partita un peso usato nella likelihood.

**Task:**
- **Peso temporale:** `w_time = exp(-ξ · giorni_fa)`, con `ξ` da config (default emivita 2 anni).
- **Peso competizione:** mappare `tournament` → moltiplicatore (sezione 1). Tornei non mappati →
  default conservativo (es. 0.6) e loggare i nomi non mappati per revisione.
- Peso finale `w = w_time · w_comp`.
- Funzione `compute_weights(df, reference_date, xi)` (la `reference_date` deve essere parametrica:
  serve per la validazione temporale, dove "oggi" è una data nel passato).

**Definition of Done:**
- [ ] `compute_weights` testata: una partita di oggi pesa ~1·w_comp; una a 1 emivita pesa ~0.5·w_comp.
- [ ] La `reference_date` è iniettabile (NO `datetime.now()` hardcodato).

---

## ISSUE #4 — Likelihood Dixon-Coles
**Milestone:** M2 · **Dipende da:** #3

**Obiettivo:** funzione di (negative) log-likelihood pesata, da minimizzare.

**Modello:**
- Gol attesi: `λ = exp(α_home + β_away + γ·is_home)`, `μ = exp(α_away + β_home)`
  (γ = vantaggio casa, applicato solo se la partita NON è in campo neutro — vedi #8).
- Probabilità del singolo risultato (x gol home, y gol away):
  `P(x,y) = τ(x,y; λ,μ,ρ) · Poisson(x; λ) · Poisson(y; μ)`
- **Correzione Dixon-Coles** τ (raddrizza i punteggi bassi correlati):
  ```
  τ(0,0) = 1 − λ·μ·ρ
  τ(0,1) = 1 + λ·ρ
  τ(1,0) = 1 + μ·ρ
  τ(1,1) = 1 − ρ
  τ(x,y) = 1   altrimenti
  ```
- NLL pesata: `−Σ_i w_i · log P(x_i, y_i)` su tutte le partite.

**Task:**
- Implementare `dixon_coles_nll(params, data, weights)` vettorizzata dove possibile.
- Gestire l'**identificabilità**: vincolo sui parametri di attacco (es. media degli α = 0) per
  evitare la degenerazione del modello.

**Definition of Done:**
- [ ] La NLL è finita e differenziabile sui dati reali.
- [ ] Test su mini-dataset sintetico: con forze note, l'NLL è minima vicino ai veri parametri.
- [ ] τ implementata esattamente come sopra (test sui 4 casi speciali).

**Note:** ρ va vincolato in un range che mantenga le probabilità non-negative (tipicamente ρ piccolo, |ρ| < ~0.2).

---

## ISSUE #5 — Stima dei parametri di forza
**Milestone:** M2 · **Dipende da:** #4

**Obiettivo:** apprendere α_i, β_i (per squadra) + γ, ρ globali.

**Task:**
- `fit_model(data, weights, config)` → minimizza la NLL con `scipy.optimize.minimize`,
  metodo **L-BFGS-B**.
- Inizializzazione sensata (α, β ~ 0; γ piccolo positivo; ρ ~ 0).
- Restituire un oggetto modello con i rating per squadra accessibili e serializzabili (JSON/pickle).

**Definition of Done:**
- [ ] Convergenza in secondi/minuti su laptop (no GPU).
- [ ] I rating ordinati sono **plausibili** (le top nazionali in cima). Sanity check umano nel notebook.
- [ ] Modello salvabile/ricaricabile.

---

## ISSUE #6 — Matrice dei risultati → esiti 90'
**Milestone:** M2 · **Dipende da:** #5

**Obiettivo:** da (λ, μ) alla matrice e agli esiti dei 90'.

**Task:**
- Costruire matrice `P(x, y)` per x,y = 0..10 con correzione τ; rinormalizzare (la coda oltre 10 è trascurabile ma va gestita).
- Da lì: `P(home vince 90')` (Σ x>y), `P(pari 90')` (Σ x=y), `P(away vince 90')` (Σ x<y).
- Funzione `match_outcome_90(team_a, team_b, model)`.

**Definition of Done:**
- [ ] Le tre probabilità sommano a 1.
- [ ] Test: due squadre identiche → P(home) ≈ P(away); P(pari) ragionevole (~0.25–0.30).

---

## ISSUE #7 — Cascata 90'/supplementari/rigori → P(passa)
**Milestone:** M2 · **Dipende da:** #6

**Obiettivo:** probabilità che una squadra superi il turno.

**Logica:**
```
P(A passa) = P(A vince 90')
           + P(pari 90') · [ P(A vince ET) + P(pari ET) · P(A vince rigori) ]
```
- ET: stessa procedura della matrice ma con `λ_ET = λ/3`, `μ_ET = μ/3` (mini-matrice).
- Rigori: `P(A vince rigori)` da config (default 0.50; opz. 0.52 al favorito sui 90').

**Task:**
- Funzione `advance_probability(team_a, team_b, model)` → `P(A passa)` (e `P(B passa) = 1 − P(A passa)`).

**Definition of Done:**
- [ ] `P(A passa) + P(B passa) = 1`.
- [ ] La squadra più forte ha P(passa) > 0.5 e resta più forte anche nel ramo ET (eredita i λ).
- [ ] Test sui casi limite (forze uguali → 0.5).

---

## ISSUE #8 — Vantaggio casa e logica host 2026
**Milestone:** M2 · **Dipende da:** #5

**Obiettivo:** stimare γ dallo storico e applicarlo correttamente al 2026.

**Task:**
- γ stimato già nel fit (#5) usando il flag `neutral`: vantaggio applicato solo a partite **non** neutre.
- Per il 2026 (tri-ospitante USA/Canada/Messico), implementare la policy:
  - γ **pieno** se la nazione gioca nel proprio paese ospitante;
  - γ **ridotto (es. ½)** se un host gioca in un altro dei tre paesi ospitanti;
  - γ = 0 per tutte le altre.
- Questa policy è una **scelta di giudizio non validabile** (nessun precedente di Mondiale tri-host):
  parametrizzarla in config e dichiararla come assunzione nel README.

**Definition of Done:**
- [ ] γ stimato è positivo e di ordine plausibile (~+0.3/+0.4 in log-gol storicamente).
- [ ] Funzione che, dato l'incontro e la sede, applica il γ corretto secondo la policy host.

---

## ISSUE #9 — Framework di validazione temporale
**Milestone:** M3 · **Dipende da:** #5

**Obiettivo:** valutare il modello SOLO su partite future rispetto al training.

**Task:**
- Split per data: allena fino a una `cutoff_date`, testa sulle partite successive (es. train ≤ 2022,
  test 2023–2025). Niente shuffle.
- Walk-forward opzionale (più cutoff progressivi) per robustezza.
- **Grid search di ξ** (e poi di w in #13) scegliendo il valore che minimizza la log-loss sul test.
- Per la valutazione sul "chi passa", usare `advance_probability` come `p` predetta e l'esito reale come label.

**Definition of Done:**
- [ ] Nessuna partita di test entra mai nel training (test anti-leakage automatico).
- [ ] `reference_date` dei pesi = `cutoff_date` dello split (coerenza temporale).
- [ ] La grid search restituisce il ξ ottimo con la relativa log-loss.

---

## ISSUE #10 — Calibrazione + curva di calibrazione
**Milestone:** M3 · **Dipende da:** #9

**Obiettivo:** rendere affidabili le probabilità e DIMOSTRARLO visivamente.

**Task:**
- Applicare **isotonic regression** (o Platt) sulle probabilità out-of-sample, fittata sul test in modo
  temporalmente onesto (o con cross-validation temporale).
- Produrre il **grafico di calibrazione** (probabilità predette in bin vs frequenze osservate) +
  istogramma delle predizioni. Salvare in `notebooks/` o come PNG.

**Definition of Done:**
- [ ] Curva di calibrazione generata e leggibile.
- [ ] Confronto pre/post calibrazione documentato.
- [ ] (Questo grafico è il deliverable-prova del progetto: deve essere riproducibile da script.)

---

## ISSUE #11 — Metriche + benchmark vs closing line
**Milestone:** M3 · **Dipende da:** #10

**Obiettivo:** il "voto onesto" del modello.

**Task:**
- Calcolare **log-loss** e **Brier score** sul test (modello vs baseline "passa sempre il favorito").
- **Benchmark contro la closing line** del mercato sulle stesse partite storiche, se reperibili quote
  storiche: confrontare la log-loss del modello con quella derivata dalle quote di chiusura de-viggate
  (vedi #12 per la de-vig). Documentare il gap.
- Reportistica: tabella riassuntiva delle metriche.

**Definition of Done:**
- [ ] Log-loss e Brier del modello < baseline banale (altrimenti il modello non aggiunge nulla, da indagare).
- [ ] Se disponibili quote storiche: confronto esplicito modello vs closing line.
- [ ] **Gate M3→M4:** se la calibrazione è scarsa, fermarsi e iterare sul modello prima di toccare le scommesse.

---

## ISSUE #12 — Quote → probabilità implicite (de-vig)
**Milestone:** M4 · **Dipende da:** #11

**Obiettivo:** convertire le quote di mercato in probabilità "vere" togliendo il margine.

**Task:**
- `implied_prob(odds)` = 1/odds (grezza).
- **De-vig:** normalizzare le implicite dei due esiti così che sommino a 1 (margine rimosso).
  `p_true = implied_i / Σ implied`. (Per la fase a eliminazione l'esito è binario passa/non-passa,
  quindi due quote.)
- Funzione `devig_two_way(odds_a, odds_b)` → (p_a, p_b).

**Definition of Done:**
- [ ] Le probabilità de-viggate sommano a 1.
- [ ] Test sull'esempio: odds 1.67/2.10 → ~55.7%/44.3%.

---

## ISSUE #13 — Blending col mercato (doppio output)
**Milestone:** M4 · **Dipende da:** #12

**Obiettivo:** due output distinti con finalità diverse.

**Task:**
- **Output A (calibrato):** `p_A = w · p_mercato + (1−w) · p_modello`, con `w` alto, scelto per
  minimizzare la log-loss out-of-sample (grid in #9). È l'output da mostrare/usare per la prova di qualità.
- **Output B (modello puro):** `w = 0`. Usato SOLO per misurare lo scarto col mercato (l'edge), non
  per "essere calibrati".
- Tenere i due output separati e documentati: servono a obiettivi opposti (calibrazione vs edge).

**Definition of Done:**
- [ ] Entrambi gli output calcolabili per ogni partita.
- [ ] `w` di A selezionato per validazione, non a mano.
- [ ] README chiarisce: A = affidabilità, B = ricerca dell'edge.

---

## ISSUE #14 — Edge + Kelly frazionato + soglia
**Milestone:** M4 · **Dipende da:** #13

**Obiettivo:** dato (p_modello, quota), decidere se e quanto puntare.

**Logica:**
```
edge (EV per unità) = p · quota − 1
b = quota − 1 ;  q = 1 − p
kelly_full = (b·p − q) / b          # = edge / b
kelly_frac = kelly_full / 4         # Kelly 1/4
punta = (edge >= soglia_min)        # soglia da config (default 0.03–0.04)
importo = kelly_frac · bankroll     # solo se punta == True e kelly_frac > 0
```

**Task:**
- Funzione `bet_decision(p_model, odds, bankroll, config)` → `{edge, punta: bool, importo}`.
- Imporre la **soglia minima di edge** (sotto soglia = rumore di stima, non si punta).
- Output negativi/zero di Kelly → niente puntata.

**Definition of Done:**
- [ ] Test sull'esempio: p=0.62, quota=1.67 → edge ≈ +0.035, kelly_full ≈ 0.053, kelly_frac ≈ 0.013.
- [ ] Edge sotto soglia → `punta = False`.
- [ ] Nessuna puntata con Kelly ≤ 0.

**Nota didattica (mettere nel README):** l'edge ha senso SOLO se `p` è calibrata (Issue #10–#11).
Su `p` scalibrata, l'edge è una fantasia. Puntare solo cifre che si è sereni di perdere.

---

## ISSUE #15 — Tracking del CLV (Closing Line Value)
**Milestone:** M4 · **Dipende da:** #14

**Obiettivo:** misurare il vero "scoreboard" delle scommesse — non il vinto/perso.

**Task:**
- Per ogni scommessa registrare: data, partita, `p_modello`, **quota di entrata** (linea morbida, presto),
  **quota di chiusura** (closing line), esito reale.
- Calcolare il **CLV**: quanto la quota si è mossa a favore tra entrata e chiusura
  (es. `CLV% = (quota_entrata / quota_chiusura) − 1`, o l'equivalente in probabilità de-viggate).
- Metriche aggregate: CLV medio, % scommesse con CLV positivo. (Il P&L si traccia ma con l'avvertenza
  che è rumorosissimo: il CLV accumula segnale molto più in fretta.)
- Persistenza: CSV/parquet del registro scommesse.

**Definition of Done:**
- [ ] Registro scommesse persistente e ispezionabile.
- [ ] CLV calcolato per scommessa + aggregato.
- [ ] README chiarisce: si scommette **presto** (linea morbida); la closing line serve **dopo**, come
  prova di aver avuto ragione. NON si aspetta la chiusura per puntare.

---

## ISSUE #16 — Monte Carlo del tabellone
**Milestone:** M5 · **Dipende da:** #7

**Obiettivo:** propagare le P per-partita lungo il bracket per le % di avanzamento/vittoria torneo.

**Task:**
- Definire il tabellone a eliminazione (struttura del bracket 2026, fase KO).
- Simulare N volte (default 50.000) campionando ogni turno con `advance_probability`; **vettorizzare con NumPy**.
- Output: per ogni squadra, P(raggiunge quarti/semi/finale/vince).

**Definition of Done:**
- [ ] Le probabilità di vittoria torneo sommano a ~1 (entro il rumore MC).
- [ ] Passare da 50k a 500k stringe gli intervalli MC ma NON cambia sistematicamente le stime
      (verifica del Principio #5: il MC propaga, non crea).
- [ ] Esecuzione in pochi secondi.

**Nota:** il MC è un megafono. Se le P per-partita sono buone, le % torneo sono buone e spettacolari;
se sono cattive, il MC le rende cattive con grande precisione. Non aggiunge affidabilità.

---

## ISSUE #17 — (Backlog) Feature esogene
**Milestone:** M6 · **Dipende da:** M3 solida

**Obiettivo:** migliorare la stima nelle fasi avanzate, dove il modello base si degrada.

**Task (in ordine di rapporto valore/sforzo):**
- **Valore rosa (Transfermarkt)** come prior per lo **shrinkage** sulle squadre con pochi dati
  (es. neopromosse con poche partite recenti) → versione bayesiana/gerarchica.
- (Se i dati esistono) **xG** al posto dei gol grezzi — ATTENZIONE: per le nazionali sono scarsi e
  disomogenei. Non costruire la pipeline core attorno a una feature presente solo per metà squadre.
- Forma recente pesata per forza avversari; giorni di riposo; assenze titolari (feature "torneo-time").

**Definition of Done:**
- [ ] Ogni feature aggiunta è valutata col confronto log-loss out-of-sample (deve migliorare, non solo "esserci").

---

## ISSUE #18 — (Backlog) Ensemble
**Milestone:** M6 · **Dipende da:** #17

**Obiettivo:** combinare modelli che sbagliano in modi diversi.

**Task:**
- Aggiungere **Elo** (regola di update online, naturale per l'aggiornamento sui risultati) e/o
  **gradient boosting (XGBoost)** sulle feature.
- Media/blend dei modelli, pesi scelti per validazione temporale.

**Definition of Done:**
- [ ] L'ensemble migliora la log-loss out-of-sample rispetto al solo Dixon-Coles.

---

## 4. Promemoria finale per chi implementa

- Prima la **pipeline onesta v1**: dati → Dixon-Coles → cascata → validazione temporale → calibrazione →
  (mercato/edge) → Monte Carlo. Feature ed ensemble DOPO.
- Il **gate M3** è sacro: niente scommesse senza calibrazione dimostrata.
- Ogni miglioria si giudica con la **log-loss out-of-sample**, mai con "sembra meglio" o col risultato
  di poche partite.
- Inquadramento **didattico**: il deliverable di valore è la curva di calibrazione + il tracking del CLV,
  non un profitto.