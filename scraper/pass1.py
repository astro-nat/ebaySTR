import asyncio
import httpx
import pandas as pd
import re
import json
import os
from typing import List, Dict

class Phase1Scraper:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        
        self.base_url = self.config["api"]["base_url"]
        self.timeout = self.config["api"]["timeout_seconds"]
        self.headers = {
            "User-Agent": self.config["api"]["user_agent"],
            "Accept": "application/json",
            "Referer": "https://hibid.com/"
        }
        
        self.zip_code = self.config["sourcing"]["zip_code"]
        self.radius = self.config["sourcing"]["radius_miles"]
        self.page_size = self.config["sourcing"]["page_size"]
        
        self.ship_killers = self.config["logistics"]["ship_killers"]
        self.mailbox_winners = self.config["logistics"]["mailbox_winners"]

    def _load_config(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found at: {filepath}")
        with open(filepath, 'r') as file:
            return json.load(file)

    def classify_logistics(self, title: str, category: str) -> str:
        text = f"{title} {category}".lower()
        if re.search(self.ship_killers, text):
            return "HARD"
        if re.search(self.mailbox_winners, text):
            return "EASY"
        return "NEUTRAL"

    async def fetch_local_auctions(self, client: httpx.AsyncClient) -> List[Dict]:
        url = f"{self.base_url}/event/list"
        params = {
            "zip": self.zip_code,
            "miles": self.radius,
            "status": "open",
            "pageSize": 50
        }
        
        try:
            response = await client.get(url, headers=self.headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            events = response.json().get('events', [])
            
            return [
                {
                    "auction_id": e.get('id'),
                    "name": e.get('eventName'),
                    "distance": e.get('distance'),
                    "shipping_offered": e.get('shippingOffered')
                }
                for e in events 
                if e.get('distance', 999) <= self.radius or e.get('shippingOffered')
            ]
        except Exception as e:
            print(f"Error fetching auctions: {e}")
            return []

    async def fetch_lots_for_auction(self, client: httpx.AsyncClient, auction_id: str) -> List[Dict]:
        url = f"{self.base_url}/lot/list"
        params = {
            "auctionId": auction_id,
            "page": 1,
            "pageSize": self.page_size
        }
        
        try:
            response = await client.get(url, headers=self.headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            lots = response.json().get('lots', [])
            
            processed_lots = []
            for lot in lots:
                title = lot.get('title', '')
                category = lot.get('lotGroupTitle', '')
                
                processed_lots.append({
                    "lot_id": lot.get('id'),
                    "auction_id": auction_id,
                    "title": title,
                    "category": category,
                    "current_bid": lot.get('currentBidAmount', 0.0),
                    "bid_count": lot.get('bidCount', 0),
                    "logistics_ease": self.classify_logistics(title, category)
                })
            return processed_lots
        except Exception as e:
            print(f"Error fetching lots for {auction_id}: {e}")
            return []

    async def run(self) -> pd.DataFrame:
        async with httpx.AsyncClient() as client:
            auctions = await self.fetch_local_auctions(client)
            if not auctions:
                return pd.DataFrame()

            tasks = [self.fetch_lots_for_auction(client, a['auction_id']) for a in auctions]
            results = await asyncio.gather(*tasks)
            
            all_lots = [item for sublist in results for item in sublist]
            df = pd.DataFrame(all_lots)
            
            if not df.empty:
                df = df[df['logistics_ease'] != "HARD"]
            return df