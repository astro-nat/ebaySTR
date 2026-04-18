import re
import pandas as pd
from transformers import pipeline
import warnings

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


class Phase2Scraper:
    def __init__(self):
        print("Initializing NLP Engine (this may take a moment if downloading the model)...")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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
            # Skip if every word already in title
            words = set(re.findall(r'[a-z0-9]+', low))
            if words and words.issubset(title_words):
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

    def batch_audit(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """Run AI condition audit on a DataFrame that has a 'description' column.

        Enriches titles using description details and runs condition classification.

        Args:
            df: DataFrame with at least 'title' and 'description' columns
            progress_callback: Optional callable(current, total) for progress updates

        Returns:
            Original DataFrame with 'enriched_title', 'verdict', 'confidence',
            and 'red_flag' columns added.
        """
        enriched_titles = []
        verdicts = []
        confidences = []
        red_flags = []

        total = len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            title = row.get('title', '')
            desc = row.get('description', '')

            # Enrich title from description
            enriched_titles.append(self._enrich_title(title, desc))

            # Condition audit
            result = self.analyze_condition(desc)
            verdicts.append(result['verdict'])
            confidences.append(result['confidence'])
            red_flags.append(result['red_flag'])

            if progress_callback:
                progress_callback(i + 1, total)

        df = df.copy()
        df['enriched_title'] = enriched_titles
        df['verdict'] = verdicts
        df['confidence'] = confidences
        df['red_flag'] = red_flags
        return df