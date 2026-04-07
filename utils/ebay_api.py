import os
import statistics
from ebaysdk.finding import Connection as Finding
from ebaysdk.exception import ConnectionError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EbayClient:
    def __init__(self):
        """
        Initializes the eBay API client using credentials from the .env file.
        The Finding API only requires the App ID (Client ID), keeping the auth flow fast.
        """
        self.app_id = os.getenv("EBAY_CLIENT_ID")
        
        if not self.app_id:
            raise ValueError("CRITICAL: EBAY_CLIENT_ID is missing from the .env file.")

        # Initialize the Finding API connection
        self.api = Finding(appid=self.app_id, config_file=None)

    def get_market_data(self, clean_title: str) -> dict:
        """
        Fetches market intelligence for a sanitized title.
        Returns the metrics required for the ROI and DTS calculations.
        """
        if not clean_title:
            return {"median_sold": 0.0, "sold_count": 0, "active_count": 0, "price_variance": 0.0}

        try:
            # 1. Fetch Sold Listings (The "Demand")
            solds = self._fetch_solds(clean_title)
            
            # Extract prices, defaulting to 0.0 if the API payload is empty
            prices = []
            for item in solds:
                try:
                    price = float(item.sellingStatus.currentPrice.value)
                    prices.append(price)
                except AttributeError:
                    continue

            # Calculate the Median (more resilient to outliers than the Mean)
            median_price = statistics.median(prices) if prices else 0.0
            variance = statistics.stdev(prices) if len(prices) > 1 else 0.0

            # 2. Fetch Active Listings (The "Supply")
            actives = self._fetch_actives(clean_title)
            active_count = len(actives)

            return {
                "median_sold": round(median_price, 2),
                "sold_count": len(prices),
                "active_count": active_count,
                "price_variance": round(variance, 2)
            }

        except ConnectionError as e:
            print(f"eBay API Error for '{clean_title}': {e}")
            return {"median_sold": 0.0, "sold_count": 0, "active_count": 0, "price_variance": 0.0}

    def _fetch_solds(self, keyword: str):
        """
        Queries the API for completed, sold items over the last 90 days.
        Filters out 'For Parts/Not Working' condition (Condition ID: 7000).
        """
        request = {
            'keywords': keyword,
            'itemFilter': [
                {'name': 'Condition', 'value': ['1000', '2000', '3000', '4000']}, 
                {'name': 'SoldItemsOnly', 'value': 'true'}
            ],
            'paginationInput': {'entriesPerPage': 50}
        }
        
        response = self.api.execute('findCompletedItems', request)
        if response.reply.ack == "Success":
            # ebaysdk returns a dictionary-like object; we safely extract the item list
            return getattr(response.reply.searchResult, 'item', [])
        return []

    def _fetch_actives(self, keyword: str):
        """
        Queries the API for currently active items to gauge market saturation.
        """
        request = {
            'keywords': keyword,
            'itemFilter': [
                {'name': 'Condition', 'value': ['1000', '2000', '3000', '4000']},
                {'name': 'ListingType', 'value': ['FixedPrice', 'AuctionWithBIN']}
            ],
            'paginationInput': {'entriesPerPage': 50}
        }
        
        response = self.api.execute('findItemsAdvanced', request)
        if response.reply.ack == "Success":
            return getattr(response.reply.searchResult, 'item', [])
        return []

# --- Local Test Execution ---
if __name__ == "__main__":
    # To test this locally: 
    # 1. Add your EBAY_CLIENT_ID to the .env file.
    # 2. Run: python -m utils.ebay_api
    
    try:
        client = EbayClient()
        test_keyword = "Nikon FE 35mm Camera Body"
        
        print(f"Fetching market data for: '{test_keyword}'...")
        market_data = client.get_market_data(test_keyword)
        
        print("\n--- Market Intelligence Report ---")
        print(f"Median Sold Price: ${market_data['median_sold']}")
        print(f"Total Solds (90 Days): {market_data['sold_count']}")
        print(f"Active Listings: {market_data['active_count']}")
        print(f"Price Volatility (StDev): ${market_data['price_variance']}")
        
    except ValueError as e:
        print(e)