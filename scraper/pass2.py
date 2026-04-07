import httpx
import re
from transformers import pipeline
import warnings

# Suppress HuggingFace warnings for cleaner terminal output
warnings.filterwarnings("ignore")

class Phase2Scraper:
    def __init__(self):
        # Initialize the zero-shot classification pipeline
        print("Initializing NLP Engine (this may take a moment if downloading the model)...")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Define the categories we want the AI to look for
        self.risk_labels = [
            "broken, damaged, or for parts",
            "untested or unknown condition",
            "mint condition or working perfectly",
            "normal wear and tear"
        ]

    async def fetch_item_description(self, client: httpx.AsyncClient, lot_id: str) -> str:
        """Fetches the full HTML description for a specific lot."""
        # HiBid API endpoint for specific lot details
        url = f"https://hibid.com/api/v1/lot/{lot_id}"
        
        try:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            # Extract the raw description
            description = data.get('description', '')
            
            # Strip HTML tags for clean NLP processing
            clean_text = re.sub('<[^<]+?>', ' ', description)
            return clean_text.strip()
            
        except Exception as e:
            print(f"Error fetching description for Lot {lot_id}: {e}")
            return ""

    def analyze_condition(self, description_text: str) -> dict:
        """Runs the HuggingFace model against the text."""
        if not description_text or len(description_text) < 10:
            return {"verdict": "Unknown", "confidence": 0.0, "red_flag": False}

        # Truncate text if it's excessively long to save compute time
        text_to_analyze = description_text[:1000] 

        # Perform zero-shot classification
        result = self.classifier(text_to_analyze, self.risk_labels)
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        # Determine if the top label indicates a risky investment
        is_red_flag = top_label in ["broken, damaged, or for parts", "untested or unknown condition"]

        return {
            "verdict": top_label,
            "confidence": round(top_score * 100, 1),
            "red_flag": is_red_flag
        }

    async def run_audit(self, lot_id: str) -> dict:
        """Executes the full Pass 2 process for a single item."""
        async with httpx.AsyncClient() as client:
            description = await self.fetch_item_description(client, lot_id)
            analysis = self.analyze_condition(description)
            return analysis