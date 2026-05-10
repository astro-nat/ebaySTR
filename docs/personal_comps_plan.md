# Personal Comps Intelligence — Build Plan

**Goal:** Replace pure eBay-scraped comps with a hybrid system that learns from YOUR actual buy/sell outcomes. Three milestones, each independently shippable.

**Why this beats fine-tuning an LLM:** Comps are time-bound numbers. Fine-tuned weights bake facts as of training day; comps go stale in weeks. RAG + a small calibration regression updates daily, costs ~$0 to run, and explains its predictions ("here are 8 similar lots I sold for $X-$Y") which fine-tuned LLMs can't.

---

## Phase 0 — Decisions before any code

Pick these once. They shape the schema.

| Decision | Recommendation | Why |
|---|---|---|
| **Storage** | SQLite (`.cache/personal_comps.db`) | Single-file, no daemon, `pandas.read_sql` works, supports `sqlite-vec` for embeddings |
| **Embedding model** | OpenAI `text-embedding-3-small` (1536-dim, $0.02/1M tokens) | Cheap, fast, beats local models on short-text similarity. Switch to `voyage-2` later if needed |
| **Embedding source** | OpenAI API | Local Ollama embeddings work but add ~200ms/call vs ~50ms |
| **What's an "outcome"** | A row that has BOTH `paid_amount` AND `sold_at_price` (or `marked_unsold`) | Drives the calibration model |
| **Schema philosophy** | Append-only, never UPDATE rows. Treat events as immutable history | Lets you replay decision history; never lose data to a bug |
| **Privacy** | Local SQLite, never uploaded. Embeddings called on titles only (no PII) | Simple |

---

## Phase 1 — Outcomes table (Week 1, the critical foundation)

Without this, nothing else works. Calibration needs ground truth.

### 1.1 New file: `scraper/outcomes.py`

Public API:
```python
record_purchase(lot_id, paid_amount, won_at, fees_paid, ship_cost, notes)
record_listing(lot_id, listed_price, listed_at, listed_url)
record_sale(lot_id, sold_price, sold_at, sold_to_platform, fees_taken)
record_unsold(lot_id, removed_at, reason)         # gave up on it
get_outcome(lot_id) -> dict | None
list_outcomes(min_paid=None, only_sold=False) -> list[dict]
calibration_pairs() -> list[(predicted_resale, actual_sale, brand, category)]
```

### 1.2 Schema (`outcomes` table in `personal_comps.db`)

```sql
CREATE TABLE outcomes (
  lot_id            TEXT NOT NULL,
  event_type        TEXT NOT NULL,    -- 'purchase' | 'listing' | 'sale' | 'unsold'
  event_ts          TEXT NOT NULL,    -- ISO 8601, when event happened in real life
  recorded_at       TEXT NOT NULL,    -- ISO 8601, when YOU logged it (audit trail)
  amount            REAL,             -- paid_amount / listed_price / sold_price
  fees              REAL,             -- buyer's premium + ship for purchase, eBay take for sale
  platform          TEXT,             -- 'ebay' | 'mercari' | 'depop' | 'local'
  url               TEXT,             -- listing URL when applicable
  notes             TEXT,
  -- Snapshot of the prediction AT DECISION TIME (frozen, never updated):
  predicted_resale  REAL,
  predicted_profit  REAL,
  brand             TEXT,
  category          TEXT,
  bolo_tier         INTEGER,
  PRIMARY KEY (lot_id, event_type, event_ts)
);
CREATE INDEX outcomes_lot ON outcomes(lot_id);
CREATE INDEX outcomes_brand_cat ON outcomes(brand, category);
CREATE INDEX outcomes_event ON outcomes(event_type, event_ts);
```

**Key insight:** when a `purchase` row is written, it captures the predicted_resale **as it was at decision time**, frozen. Later when `sale` is logged, you have a clean (predicted, actual) pair for calibration — no need to remember "what did the model say back then."

### 1.3 UI integration (modifications to `app.py`)

Three new buttons on the analysis view:
```
[💰 I won this — record purchase]   → modal: paid_amount, fees, notes
[📦 Listed it — record listing]     → modal: platform, listed_price, URL
[💵 Sold!]                          → modal: sold_price, fees_taken, notes
```

Plus a new sidebar tab: **📊 Outcomes** — table view of every (purchase, listing, sale) tuple with derived columns: `actual_profit`, `realized_vs_predicted`, `days_to_sell`.

### 1.4 Acceptance criteria

- Can record a buy → list → sale flow end-to-end
- `outcomes.calibration_pairs()` returns N rows after N completed (purchase + sale) cycles
- Outcomes table shows per-row `predicted_profit` (frozen from decision day) vs `actual_profit` (computed from purchase + sale rows)

**Estimated effort: 4-6 hours.** Hardest part is the modal UX, not the data layer.

---

## Phase 2 — Embeddings & similarity search (Week 4-6, after ~50 outcome rows)

Don't build until Phase 1 has data flowing. Embeddings without outcomes is just fancy search.

### 2.1 New file: `scraper/embeddings.py`

```python
embed_texts(texts: list[str]) -> np.ndarray  # batched, cached
embed_lot(lot_id, title, brand, category, description) -> None  # idempotent
nearest_lots(query_text, k=10, brand=None) -> list[(lot_id, similarity, metadata)]
embed_backlog() -> int  # one-shot: embed every lot in outcomes that lacks an embedding
```

### 2.2 New SQLite tables (same `personal_comps.db`)

```sql
-- Vector index, requires sqlite-vec extension
CREATE VIRTUAL TABLE lot_embeddings USING vec0(
  lot_id     TEXT PRIMARY KEY,
  embedding  FLOAT[1536]
);

-- Side table for the metadata you'll filter on
CREATE TABLE lot_metadata (
  lot_id      TEXT PRIMARY KEY,
  title       TEXT,
  brand       TEXT,
  category    TEXT,
  embedded_at TEXT
);
```

### 2.3 Cost & batch sizing

- `text-embedding-3-small` at $0.02/1M tokens = **$0.0001 per lot** (typical title is ~50 tokens)
- 1,000 lots = $0.10 one-time
- Batch 100 at a time (one API call per batch, ~50ms each)

### 2.4 New file: `scripts/embed_backlog.py`

One-shot script that walks `comped_lots.json` + `outcomes.db` and embeds anything missing. Idempotent. Run periodically.

### 2.5 UI integration

New panel on the analysis view, **above** the comps table:

```
🔍 Similar lots in YOUR history (Phase 2)
─────────────────────────────────────────
Showing 5 lots most similar to "Webkinz Magical Retriever HM779"
sorted by embedding similarity:

  [0.92] Webkinz Love Puppy HM131 (won 2026-03-14, paid $42, sold $187, +$94 profit)
  [0.87] Webkinz Cheeky Cat HM064 (won 2025-12-02, paid $58, sold $230, +$118)
  [0.84] Webkinz Sugar Plum Pegasus (lost — outbid at $89)
  [0.81] Webkinz Pink Glitter Pony (won 2026-01-20, listed $135, NOT YET SOLD)
  [0.79] Webkinz Strawberry Cloud (won 2026-02-08, paid $25, sold $185, +$140)

  → For 4 wins, your average realization: 0.79× the eBay-comp prediction.
  → For "predicted $200 resale" → expect actual ~$158.
```

This is the **killer feature** that fine-tuned LLMs can't compete with: it's not a prediction, it's evidence.

**Estimated effort: 6-8 hours.** Most of it is the UI render + similarity-search query.

---

## Phase 3 — Calibration model (Week 8+, when ≥100 sold outcomes exist)

Below 100 rows the model is overfit garbage. Wait for the data.

### 3.1 New file: `scraper/calibration.py`

```python
fit_calibration(min_samples=100) -> dict[str, Any]    # returns model + metadata
load_calibration() -> CalibrationModel | None
predict_actual(predicted_resale, brand, category, ...) -> tuple[float, float]
                                                        # (predicted_actual, confidence_interval)
recalibrate_if_stale(threshold_days=14) -> bool        # auto-refit when new outcomes arrive
```

### 3.2 Model choice

`sklearn.ensemble.GradientBoostingRegressor`, target = `actual_sold_price`. Features:
- `predicted_resale` (eBay median)
- `comp_count` (how many comps were found)
- `bolo_tier` (1 / 2 / 3)
- `brand_one_hot` (top 30 brands one-hot, rest as 'Other')
- `category_one_hot` (top 15)
- `days_to_sell_pct` (median for similar items, time-of-year proxy)
- Optional: `condition_score` if you start logging it

### 3.3 Storage

Pickle to `.cache/calibration_model.pkl`. Refit weekly or on-demand from `outcomes.db`. Store training metadata (sample count, fit timestamp, validation MAE) so the UI can warn when the model is stale.

### 3.4 UI integration

In the analysis view, **next to** every `Est. Resale` and `Est. Profit` cell:

```
Est. Resale (median):  $187   [eBay]
                       $148   [Your history × 0.79 calibration · n=23 similar]
                                                      ↑ click to see how

Est. Profit:           $112   [eBay-based]
                       $79    [Your-history-calibrated]   ⚠ 30% gap — trust history
```

When the two predictions disagree by >30%, surface a warning. When they agree, show one number with high confidence.

### 3.5 Calibration dashboard tab

New sidebar tab: **🎯 Calibration**:
- Scatter: `predicted_resale vs actual_sale` (one dot per completed cycle)
- Per-brand realization rate table
- Per-category realization rate table
- Time-of-year pattern (Q4 holiday lift, summer slump)
- Honest assessment: "Your model has 23 sold outcomes. It's reliable for [Pyrex, Sterling silver]. It's NOT YET RELIABLE for [Tamagotchi, Gymboree] — too few examples."

**Estimated effort: 8-12 hours,** spread over multiple sessions as data accumulates.

---

## Phase 4 (optional) — RAG with Claude/GPT-4

You may never need this. Phases 1-3 cover 90% of the value. Build only if calibration is missing qualitative signals (condition deltas, era authentication).

If built: pull top-K similar lots from Phase 2's similarity search, format as context, ask Claude "given these 8 similar items I sold for [X, Y, Z], predict this new lot's actual sell price." Cost: ~$0.005 per prediction. Not for high-volume scans, just deep-dive on $200+ lots.

---

## Files & integration map

```
scraper/
  outcomes.py              [NEW — Phase 1]
  embeddings.py            [NEW — Phase 2]
  calibration.py           [NEW — Phase 3]
  my_bids.py               [no change — already tracks bids; outcomes.py joins on lot_id]
  comped_lots.py           [no change — already has TTL'd comp registry]

scripts/
  embed_backlog.py         [NEW — one-shot embedder, Phase 2]
  fit_calibration.py       [NEW — periodic refit, Phase 3]

.cache/
  personal_comps.db        [NEW SQLite — outcomes + embeddings + lot_metadata]
  calibration_model.pkl    [NEW — fitted regressor, Phase 3]
  my_bids.json             [unchanged]
  comped_lots.json         [unchanged]

app.py
  + Outcomes tab           [Phase 1]
  + Record buttons         [Phase 1]
  + Similar-lots panel     [Phase 2]
  + Dual-prediction column [Phase 3]
  + Calibration dashboard  [Phase 3]
```

---

## Lock-in moments — when each phase becomes valuable

| Outcome rows | What you can do |
|---|---|
| 0-10 | Track outcomes — realize how broken your gut was. ($0 of value, but decoded reality.) |
| 10-50 | Spot patterns by eye in the Outcomes tab. ("Webkinz under $30 always doubles.") |
| 50-100 | Similar-lots panel becomes useful. Embedding similarity surfaces patterns you missed. |
| 100-200 | Calibration model trains. Per-brand realization rates become statistically meaningful. |
| 200+ | The model + similarity search beat eBay-median comps for YOUR sourcing patterns. Saleable as a tool. |

---

## What to do RIGHT NOW (before writing any of the above code)

For the next 3 weeks, **manually log every buy + sell outcome** in a spreadsheet or `.cache/outcomes.csv` with these columns:

```
lot_id, won_date, paid, predicted_resale_at_decision, listed_price, sold_date, sold_for, platform
```

After 3 weeks of manual logging, two things will be true:

1. You'll have 20-50 rows of ground truth — **enough to know if the existing eBay comps are over- or under-predicting** (you don't need a model to see this)
2. You'll know exactly which fields you wished were captured — informs the schema before you commit code

THEN come back and build Phase 1. The schema you design with 30 outcomes in hand will be 10× better than the schema you'd design today.

---

## Tools / libraries to use (and avoid)

**Use:**
- `sqlite-vec` (vector index in a single SQLite file)
- `voyage-2` or `text-embedding-3-small` (embeddings)
- `scikit-learn` (regression — `pip install`, no infra)
- `litellm` (one client that talks to Claude / GPT-4 / local Ollama interchangeably — keeps you provider-agnostic)

**Don't use:**
- Pinecone / Weaviate / Chroma — overkill for <100k items, adds infra
- Fine-tuning anything — wrong tool, see top of doc

---

## Status tracker

| Phase | Status | Started | Outcome rows at start | Notes |
|---|---|---|---|---|
| Phase 0: Decisions | ☐ Done | — | 0 | Pick storage, embedding model |
| Phase 1: Outcomes table | ☐ Not started | — | — | Wait until manual logging shows ~20 rows |
| Phase 2: Embeddings | ☐ Not started | — | — | Need ≥50 outcomes first |
| Phase 3: Calibration | ☐ Not started | — | — | Need ≥100 sold outcomes |
| Phase 4: RAG (optional) | ☐ Not started | — | — | Only if Phase 3 is insufficient |

Update this table as you progress. Document any deviations from the plan above with a short "decided to do X instead because Y" note.
