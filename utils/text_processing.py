import re
from rapidfuzz import fuzz

class TitleSanitizer:
    def __init__(self):
        """
        Initializes the TitleSanitizer and compiles the regular expressions.
        Compiling regex upfront improves processing speed when looping through hundreds of lots.
        """
        # Matches common auction house "fluff" and condition notes that break API searches.
        self.noise_pattern = re.compile(
            r'(?i)\b(l@@k|rare|vintage|antique|lot of|estate find|must see|'
            r'nib|nos|mint|used|as-is|as is|untested|works|great condition)\b'
        )
        
        # Matches typical lot numbering formats at the beginning of a string.
        # Examples: "Lot 123", "#45", "Item 12:", "Lot No. 5"
        self.lot_number_pattern = re.compile(r'(?i)^(lot|item)?\s*(no\.?)?\s*#?\s*\d+[\s-:]*')

    def sanitize_for_ebay(self, raw_title: str) -> str:
        """
        Cleans a raw auction title to maximize eBay Browse/Finding API match rates.
        """
        if not raw_title or not isinstance(raw_title, str):
            return ""

        # 1. Strip leading lot numbers
        clean_title = self.lot_number_pattern.sub('', raw_title)
        
        # 2. Remove auction fluff keywords
        clean_title = self.noise_pattern.sub('', clean_title)
        
        # 3. Remove special characters (keep alphanumeric and spaces)
        clean_title = re.sub(r'[^\w\s-]', ' ', clean_title)
        
        # 4. Normalize whitespace (remove double spaces caused by the deletions above)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        
        # Fallback: if the cleaning somehow stripped everything, return the alphanumeric original
        if not clean_title:
            return re.sub(r'[^\w\s]', '', raw_title).strip()
            
        return clean_title

    def calculate_match_confidence(self, hibid_title: str, ebay_title: str) -> float:
        """
        Uses RapidFuzz to score the similarity between our cleaned HiBid title 
        and the returned eBay title. This prevents false-positive market comps.
        """
        if not hibid_title or not ebay_title:
            return 0.0
            
        # token_sort_ratio ignores word order, which is perfect for eBay titles
        return fuzz.token_sort_ratio(hibid_title.lower(), ebay_title.lower())

# --- Local Test Execution ---
if __name__ == "__main__":
    sanitizer = TitleSanitizer()
    
    test_titles = [
        "Lot #42 - VINTAGE Nikon FE 35mm Camera Body AS-IS L@@K",
        "14k Gold Herringbone Chain 5g Estate Find",
        "Item 005: Nintendo Gameboy Color Teal Works Great",
        "#112 - RARE Apple iPad Pro 4th Gen UNTESTED"
    ]
    
    print("Testing Sanitization Pipeline:\n" + "-"*40)
    for title in test_titles:
        clean = sanitizer.sanitize_for_ebay(title)
        print(f"RAW:   {title}")
        print(f"CLEAN: {clean}")
        
        # Simulating an eBay API return to test the RapidFuzz logic
        mock_ebay_return = clean + " Excellent Shape Fast Shipping"
        confidence = sanitizer.calculate_match_confidence(clean, mock_ebay_return)
        print(f"MATCH CONFIDENCE: {confidence:.2f}%\n")