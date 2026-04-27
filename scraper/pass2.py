import re
import pandas as pd
from transformers import pipeline
import warnings

# Re-check pickup-only language at audit time in case the auction was loaded
# from cache (where logistics_ease was computed with an older, narrower regex).
# Keep this list in sync with Phase1Scraper._PICKUP_ONLY_RE.
_PICKUP_ONLY_AUDIT_RE = re.compile(
    r'local\s+pick\s*-?\s*up\s+only'
    r'|pick\s*-?\s*up\s+only'
    r'|local\s+pickup\s+only'
    r'|pickup\s+only'
    r'|in[- ]?(store|person)\s+pick\s*-?\s*up\s+only'
    r'|in[- ]?(store|person)\s+only'
    r'|on[- ]?site\s+pick\s*-?\s*up'
    r'|must\s+pick\s*-?\s*up'
    r'|must\s+be\s+picked\s+up'
    r'|buyer\s+(must\s+)?(pick\s*-?\s*up|arrange\s+pickup|arrange\s+shipp?ing|arrange\s+transport)'
    r'|pick\s*-?\s*up\s+(required|mandatory)'
    r'|no\s+shipp?ing'
    r'|will\s+not\s+ship'
    r'|do(es)?\s+not\s+ship'
    r'|cannot\s+be\s+shipped'
    r'|can\s*n?o?t\s+ship'
    r'|not\s+available\s+for\s+shipp?ing'
    r'|shipping\s+(is\s+)?not\s+available'
    r'|shipping\s*:\s*(not\s+available|none|no\b|unavailable)'
    r'|unable\s+to\s+ship'
    r'|no\s+ship\b'
    r'|local\s+(delivery|sale|buyers?)\s+only'
    r'|ships?\s+locally\s+only'
    r'|ships?\s+only\s+(locally|to\s+local)'
    r'|this\s+lot\s+(is|will\s+be)\s+(a\s+)?pick\s*-?\s*up'
    r'|available\s+for\s+pickup\s+only',
    re.IGNORECASE,
)

# Suppress HuggingFace warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# Common filler words to skip when extracting details from descriptions
_FILLER = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'this',
    'that', 'these', 'those', 'it', 'its', 'of', 'in', 'on', 'at', 'to',
    'for', 'with', 'from', 'by', 'as', 'into', 'about', 'up', 'out',
    'lot', 'item', 'items', 'listing', 'auction', 'bid', 'bidding',
    'see', 'photos', 'photo', 'pictures', 'picture', 'image', 'images',
    'please', 'note', 'description', 'details', 'condition', 'shipping',
    'sold', 'buyer', 'seller', 'payment', 'terms', 'pickup',
    'click', 'here', 'more', 'info', 'information', 'view', 'all',
    'no', 'yes', 'not', 'we', 'our', 'you', 'your', 'if', 'so',
}

# HiBid lot descriptions are littered with section-header tokens that look
# like product features to the regex extractors. These leak into enriched
# titles ("Fog Machine ... Remote Condition Very Good Damaged No In Packaging")
# and pollute the eBay search query. Treat them as zero-signal tokens.
_HIBID_HEADER_TOKENS = {
    'condition', 'damaged', 'packaging', 'package', 'packed', 'boxed',
    'new', 'used', 'good', 'fair', 'poor', 'excellent', 'mint',
    'very', 'like', 'original', 'retail', 'msrp', 'value', 'estimated',
    'est', 'worth', 'tested', 'untested', 'working', 'nonworking',
    'broken', 'parts', 'only', 'sold', 'final', 'sale', 'available',
    'included', 'includes', 'missing', 'complete', 'incomplete',
    'pickup', 'shipping', 'shipped', 'ship', 'delivery', 'local',
    'returns', 'warranty', 'guarantee', 'lot', 'unit', 'units',
}


class Phase2Scraper:
    # DEFAULT_MODEL is bart-large-mnli — the historical default, and likely
    # already cached on any machine that's run this app before, so it loads
    # instantly. DEFAULT_MODEL_FAST (DistilBART-MNLI) is ~3x faster but
    # requires a fresh ~500MB download on first use. We let the UI pick.
    DEFAULT_MODEL = "facebook/bart-large-mnli"
    DEFAULT_MODEL_FAST = "valhalla/distilbart-mnli-12-3"
    # Back-compat alias used elsewhere before the fast/accurate split
    DEFAULT_MODEL_ACCURATE = DEFAULT_MODEL

    def __init__(self, model_name: str = None):
        model = model_name or self.DEFAULT_MODEL
        self.model_name = model
        print(f"Initializing NLP Engine ({model}; may download on first use)...")
        self.classifier = pipeline("zero-shot-classification", model=model)

        self.risk_labels = [
            "broken, damaged, or for parts",
            "untested or unknown condition",
            "mint condition or working perfectly",
            "normal wear and tear"
        ]

    def _enrich_title(self, original_title: str, description: str) -> str:
        """Build a detailed, eBay-searchable title from auction title + description.

        Extracts brand names, model numbers, product specifics, and key attributes
        from the description and combines them with the original title.
        Returns the enriched title (max ~80 chars for good eBay search results).
        """
        if not description or len(description.strip()) < 10:
            return original_title

        # Strip HTML tags and normalize whitespace
        clean = re.sub(r'<[^<]+?>', ' ', description)
        clean = re.sub(r'&\w+;', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Words already in the original title (lowercase)
        title_words = set(re.findall(r'[a-z0-9]+', original_title.lower()))

        new_details = []
        seen = set()

        def _add(term):
            t = term.strip()
            if not t or len(t) < 2:
                return
            low = t.lower()
            if low in seen or low in _FILLER:
                return
            # Skip known HiBid section-header boilerplate that keeps leaking
            # into enriched titles ("Condition", "Damaged", "No In Packaging"
            # etc. are structural labels on the listing, not product features).
            if low in _HIBID_HEADER_TOKENS:
                return
            # Skip if every word of the phrase is either already in the title
            # OR a filler/header token — i.e. the phrase adds no new signal.
            # (Previously we only checked issubset(title_words), which let
            # phrases like "Remote Condition" through when the title already
            # had "Remote" and "Condition" is a header.)
            words = set(re.findall(r'[a-z0-9]+', low))
            if not words:
                return
            informative = words - title_words - _FILLER - _HIBID_HEADER_TOKENS
            if not informative:
                return
            seen.add(low)
            new_details.append(t)

        # 1. Brand / model numbers (e.g. "XR-500", "Model 42B", "HP LaserJet")
        for m in re.finditer(r'\b([A-Z][A-Za-z]*[\s-]?[A-Z0-9][\w-]*(?:[\s-][A-Z0-9][\w-]*)*)\b', clean):
            _add(m.group(1))

        # 2. Model / part numbers: alphanumeric with hyphens or dots (e.g. "A1234", "NES-001")
        for m in re.finditer(r'\b([A-Z]{1,4}[\-.]?\d{2,}[\w\-.]*)\b', clean):
            _add(m.group(1))

        # 3. Year mentions (e.g. "1943", "2019")
        for m in re.finditer(r'\b(1[89]\d{2}|20[0-2]\d)\b', clean):
            _add(m.group(1))

        # 4. Quoted product names (e.g. '"Elvis #1 Hits"')
        for m in re.finditer(r'["\u201c]([^"\u201d]{3,40})["\u201d]', clean):
            _add(m.group(1))

        # 5. Key product phrases: "Brand + Product" patterns in first 300 chars
        first_chunk = clean[:300]
        # Extract capitalized multi-word phrases (likely product names)
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', first_chunk):
            phrase = m.group(1)
            if len(phrase.split()) <= 4:
                _add(phrase)

        # 6. Extract first meaningful sentence as fallback context
        sentences = re.split(r'[.!?\n]', first_chunk)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15 and not any(skip in sent.lower() for skip in
                                           ['shipping', 'payment', 'pickup', 'bid', 'click', 'terms']):
                # Pull individual significant words from first sentence
                for word in sent.split():
                    w_clean = re.sub(r'[^a-zA-Z0-9\-]', '', word)
                    if (len(w_clean) > 3
                            and w_clean.lower() not in _FILLER
                            and w_clean[0].isupper()):
                        _add(w_clean)
                break

        # Build enriched title: original + best new details, up to ~80 chars
        enriched = original_title.rstrip('.')
        for detail in new_details:
            candidate = f"{enriched} {detail}"
            if len(candidate) > 80:
                break
            enriched = candidate

        return enriched

    def analyze_condition(self, description_text: str) -> dict:
        """Runs the HuggingFace model against a single description."""
        if not description_text or len(description_text.strip()) < 10:
            return {"verdict": "Unknown", "confidence": 0.0, "red_flag": False}

        text_to_analyze = re.sub('<[^<]+?>', ' ', description_text)[:1000]

        result = self.classifier(text_to_analyze, self.risk_labels)

        top_label = result['labels'][0]
        top_score = result['scores'][0]

        is_red_flag = top_label in [
            "broken, damaged, or for parts",
            "untested or unknown condition"
        ]

        return {
            "verdict": top_label,
            "confidence": round(top_score * 100, 1),
            "red_flag": is_red_flag
        }

    def batch_audit(self, df: pd.DataFrame, progress_callback=None,
                    batch_size: int = 8) -> pd.DataFrame:
        """Run AI condition audit on a DataFrame that has a 'description' column.

        Enriches titles using description details and runs condition
        classification. Classification runs in BATCHES rather than one text
        at a time — the HF pipeline accepts a list of texts and does a single
        forward pass per batch, amortizing Python overhead and letting PyTorch
        BLAS parallelize across cores.

        Typical speedup on CPU: 2-3x from batching alone, more when combined
        with the lighter distilbart-mnli model (~6-10x total vs serial
        bart-large).

        Args:
            df: DataFrame with at least 'title' and 'description' columns
            progress_callback: Optional callable(current, total) for progress
                updates. Fires at batch boundaries, not per-item.
            batch_size: How many descriptions to classify per forward pass.
                Memory scales with this × sequence length × label count.
                8 is a safe default on typical hardware; bump to 16 on a
                beefy machine, drop to 4 if you see OOMs.

        Returns:
            Original DataFrame with 'enriched_title', 'verdict', 'confidence',
            and 'red_flag' columns added.
        """
        total = len(df)
        titles = df['title'].fillna('').astype(str).tolist() if 'title' in df.columns else [''] * total
        descs = df['description'].fillna('').astype(str).tolist() if 'description' in df.columns else [''] * total
        logistics = (
            df['logistics_ease'].fillna('').astype(str).tolist()
            if 'logistics_ease' in df.columns else [''] * total
        )

        # --- Step 1: enrich titles (regex-only, microseconds — serial is fine) ---
        enriched_titles = [
            self._enrich_title(t, d) for t, d in zip(titles, descs)
        ]

        # --- Step 2: prep texts for classification ---
        # Skip classification for rows we can already disqualify:
        #   (a) logistics_ease == 'HARD' — Phase 1 already pattern-matched the
        #       title/category/description against known unshippable items
        #       (mattresses, vehicles, real estate, etc.). No point paying
        #       ~50ms of transformer compute to "verify" it.
        #   (b) empty/short description — would return "Unknown" anyway.
        # Skipped-HARD rows get a distinct "Unshippable (HARD logistics)"
        # verdict + red_flag=True so they're excluded from comps downstream.
        cleaned_texts: list = []
        skip_flags: list = []        # True = don't classify (for ANY reason)
        hard_flags: list = []        # True = HARD-skip specifically
        for d, log in zip(descs, logistics):
            # Re-check pickup-only against the current regex. Covers the case
            # where this auction was loaded from cache (Phase 1 regex may have
            # been narrower at the time it was scraped) and the case where
            # Phase 1's title/category match missed but the description gives
            # it away.
            pickup_only_in_desc = bool(
                d and _PICKUP_ONLY_AUDIT_RE.search(d)
            )
            is_hard = (log == 'HARD') or pickup_only_in_desc
            if is_hard:
                cleaned_texts.append("")
                skip_flags.append(True)
                hard_flags.append(True)
            elif not d or len(d.strip()) < 10:
                cleaned_texts.append("")
                skip_flags.append(True)
                hard_flags.append(False)
            else:
                cleaned_texts.append(re.sub(r'<[^<]+?>', ' ', d)[:1000])
                skip_flags.append(False)
                hard_flags.append(False)

        # Default verdict for skipped rows. HARD rows get a distinct verdict
        # so they're separable from "Unknown" (empty description) in the UI.
        verdicts: list = [
            "Unshippable (HARD logistics)" if h else "Unknown"
            for h in hard_flags
        ]
        confidences: list = [100.0 if h else 0.0 for h in hard_flags]
        red_flags: list = [bool(h) for h in hard_flags]

        # Positions to actually classify
        live_idx = [i for i, s in enumerate(skip_flags) if not s]
        live_n = len(live_idx)

        red_flag_labels = {
            "broken, damaged, or for parts",
            "untested or unknown condition",
        }

        # --- Step 3: batched classification ---
        processed = 0
        for start in range(0, live_n, batch_size):
            chunk = live_idx[start:start + batch_size]
            texts = [cleaned_texts[i] for i in chunk]

            try:
                results = self.classifier(
                    texts, self.risk_labels, batch_size=batch_size,
                )
                # Pipeline returns a single dict if given a single string, a
                # list of dicts if given a list — we always give a list here.
                if isinstance(results, dict):
                    results = [results]
            except Exception:
                # Fall back to per-item on failure so one bad row doesn't
                # torch the whole audit
                results = []
                for t in texts:
                    try:
                        results.append(self.classifier(t, self.risk_labels))
                    except Exception:
                        results.append({"labels": ["Unknown"], "scores": [0.0]})

            for pos, result in zip(chunk, results):
                top_label = result['labels'][0]
                top_score = result['scores'][0]
                verdicts[pos] = top_label
                confidences[pos] = round(top_score * 100, 1)
                red_flags[pos] = top_label in red_flag_labels

            processed += len(chunk)
            if progress_callback:
                # Report against total rows (including skipped) so the bar
                # reaches 100% — add skipped rows that fell before this
                # batch's position.
                current_total = processed + sum(
                    1 for i in range(total)
                    if skip_flags[i] and i <= (chunk[-1] if chunk else 0)
                )
                progress_callback(min(current_total, total), total)

        # Ensure final progress hit 100%
        if progress_callback:
            progress_callback(total, total)

        df = df.copy()
        df['enriched_title'] = enriched_titles
        df['verdict'] = verdicts
        df['confidence'] = confidences
        df['red_flag'] = red_flags

        # Upgrade logistics_ease to HARD for anything we re-flagged here so
        # the comps filters, results-table styling, and cache-on-save all
        # agree. Only promote (never demote) so Phase 1's EASY stays EASY.
        if 'logistics_ease' in df.columns:
            newly_hard = [
                h and (log != 'HARD')
                for h, log in zip(hard_flags, logistics)
            ]
            if any(newly_hard):
                df.loc[newly_hard, 'logistics_ease'] = 'HARD'

        # Stash diagnostics so the UI can report what got pre-filtered.
        # pandas `.attrs` survives operations that don't explicitly reset it.
        df.attrs['audit_skipped_hard'] = int(sum(hard_flags))
        df.attrs['audit_skipped_empty'] = int(
            sum(1 for s, h in zip(skip_flags, hard_flags) if s and not h)
        )
        df.attrs['audit_classified'] = int(live_n)
        df.attrs['audit_newly_hard_from_desc'] = int(
            sum(1 for h, log in zip(hard_flags, logistics) if h and log != 'HARD')
        )
        return df