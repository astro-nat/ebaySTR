"""Cross-session ScrapingBee spend ledger, keyed by lot_id.

Every lot that consumes ScrapingBee credits — whether the lookup found
comps or came back empty — is recorded here. Before any comps run, the
ledger is overlaid onto the frame: previously-priced lots get their
comp data back for free, and previously-attempted-but-empty lots are
excluded from the pending queue so we don't pay twice to learn the
same "no comps found".

Rebuilt 7/11/26 at user request ("keep data on ANYthing that used
ScrapingBee so I don't accidentally spend credits again on the same
items"). An earlier incarnation of this module was removed; the
`.cache/comped_lots.json` filename is retained so stale references
(hibid_account's snapshot builder) resolve to real data again.

Storage: one JSON file, atomic-rename writes (OneDrive-safe),
thread-safe under the comps ThreadPoolExecutor via a module lock.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_LEDGER_PATH = Path(".cache/comped_lots.json")
_LOCK = threading.Lock()
_CACHE: Optional[Dict] = None   # in-process copy, loaded once

# Columns copied into / out of the ledger per lot. `price_source`
# carries the annotations (wide-spread, off-topic dropped, short
# query) so restored rows look exactly like fresh ones.
_COMP_FIELDS = (
    'est_resale', 'price_low', 'price_high', 'comp_count',
    'ebay_comps', 'mercari_comps', 'pricecharting_comps',
    'gocollect_comps', 'price_source', 'ebay_str', 'str_source',
)


def _load() -> Dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        with open(_LEDGER_PATH, encoding='utf-8') as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or 'lots' not in data:
            data = {'lots': {}}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {'lots': {}}
    _CACHE = data
    return _CACHE


def _save(data: Dict) -> None:
    """Atomic write — temp file + rename so a mid-write crash or a
    OneDrive sync pass can't leave a truncated ledger."""
    _LEDGER_PATH.parent.mkdir(exist_ok=True)
    tmp = _LEDGER_PATH.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    os.replace(tmp, _LEDGER_PATH)


def _clean(v):
    """JSON-safe scalar: NaN/NA → None, numpy scalars → python."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, 'item'):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


def record_from_df(df: pd.DataFrame) -> int:
    """Record every row of `df` as a ScrapingBee spend event.

    Call with exactly the rows that just went through the paid lookup
    (the comps chunk) — priced or not. Returns the number recorded.
    """
    if df is None or df.empty or 'lot_id' not in df.columns:
        return 0
    now = datetime.now().isoformat()
    with _LOCK:
        data = _load()
        lots = data.setdefault('lots', {})
        n = 0
        for _, row in df.iterrows():
            lot_id = str(row.get('lot_id') or '').strip()
            if not lot_id or lot_id == 'nan':
                continue
            entry = {
                'title': str(row.get('title') or '')[:200],
                'attempted_at': now,
            }
            for f in _COMP_FIELDS:
                if f in df.columns:
                    entry[f] = _clean(row.get(f))
            lots[lot_id] = entry
            n += 1
        if n:
            _save(data)
    return n


def get(lot_id) -> Optional[Dict]:
    """Ledger entry for one lot, or None."""
    with _LOCK:
        return _load().get('lots', {}).get(str(lot_id))


def overlay_onto_df(df: pd.DataFrame, ttl_days: float = 7.0):
    """Fill comp columns from the ledger; identify re-spend blocks.

    Returns ``(df, filled_count, blocked_ids)``:
      - rows whose ledger entry (within TTL) carries a price get their
        comp columns restored — they'll no longer look "pending"
      - rows attempted within TTL that found NOTHING are returned in
        ``blocked_ids`` so the caller can exclude them from the paid
        queue (paying twice to learn "no comps" is the exact waste
        this ledger exists to stop)
    Entries older than the TTL are ignored (prices drift; re-check).
    """
    blocked_ids: set = set()
    if df is None or df.empty or 'lot_id' not in df.columns:
        return df, 0, blocked_ids
    with _LOCK:
        lots = _load().get('lots', {})
    if not lots:
        return df, 0, blocked_ids
    now = datetime.now()
    out = df.copy()
    for f in _COMP_FIELDS:
        if f not in out.columns:
            out[f] = None
    filled = 0
    for idx in out.index:
        lot_id = str(out.at[idx, 'lot_id'])
        entry = lots.get(lot_id)
        if not entry:
            continue
        # Respect TTL
        try:
            age_days = (
                now - datetime.fromisoformat(entry.get('attempted_at', ''))
            ).total_seconds() / 86400.0
        except (ValueError, TypeError):
            continue
        if age_days > ttl_days:
            continue
        # Already priced in this frame? Don't overwrite fresher data.
        try:
            if not pd.isna(out.at[idx, 'est_resale']):
                continue
        except (TypeError, ValueError):
            pass
        if entry.get('est_resale') is not None:
            for f in _COMP_FIELDS:
                if f in entry:
                    out.at[idx, f] = entry[f]
            # Annotate provenance so the user can tell ledger-restored
            # prices from fresh scrapes at a glance.
            _src = str(entry.get('price_source') or 'ledger')
            if '💾' not in _src:
                out.at[idx, 'price_source'] = f"{_src} 💾{age_days:.0f}d"
            filled += 1
        else:
            blocked_ids.add(lot_id)
    return out, filled, blocked_ids


def stats() -> Dict:
    """Ledger size summary for UI display."""
    with _LOCK:
        lots = _load().get('lots', {})
    priced = sum(1 for e in lots.values() if e.get('est_resale') is not None)
    return {
        'total': len(lots),
        'priced': priced,
        'empty_attempts': len(lots) - priced,
    }
