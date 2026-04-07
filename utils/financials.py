class FinancialEngine:
    def __init__(self, target_roi=5.0, buyers_premium=0.15, sales_tax=0.0825, platform_fee=0.15, buffer=15.0):
        """
        Initializes the financial parameters for the ROI gate.
        """
        self.target_roi = target_roi
        self.bp = buyers_premium
        self.tax = sales_tax
        self.platform_fee = platform_fee
        
        # Flat buffer for packing materials or unforeseen acquisition costs
        self.buffer = buffer 
        
        # The multiplier applied to the bid during checkout (e.g., 1.2325)
        self.acquisition_multiplier = 1 + self.bp + self.tax

    def calculate_dts(self, active_count: int, sold_count: int, timeframe_days: int = 90) -> float:
        """
        Calculates Days to Sell (DTS) using velocity and queue length.
        """
        if sold_count <= 0:
            return 999.0 # Effectively illiquid; infinite days to sell
            
        daily_velocity = sold_count / timeframe_days
        
        # We add 1 to active_count to simulate your item entering the back of the market queue
        dts = (active_count + 1) / daily_velocity
        return round(dts, 1)

    def calculate_max_bid(self, resale_value: float, logistics_penalty: float) -> float:
        """
        Calculates the absolute maximum allowable bid to hit the target ROI.
        """
        if resale_value <= 0:
            return 0.0
            
        net_proceeds = resale_value * (1 - self.platform_fee)
        allowable_cost = net_proceeds - logistics_penalty - self.buffer
        
        max_bid = allowable_cost / ((1 + self.target_roi) * self.acquisition_multiplier)
        
        # Cannot bid negative money; if it drops below zero, the item is disqualified
        return max(0.0, round(max_bid, 2))

    def evaluate_lead(self, resale_value: float, current_bid: float, logistics_penalty: float, dts: float, max_dts: float = 90.0) -> dict:
        """
        Determines if an auction lot survives the financial gate.
        """
        max_bid = self.calculate_max_bid(resale_value, logistics_penalty)
        
        # Calculate actual projected profit if won at the CURRENT bid
        total_acquisition_cost = (current_bid * self.acquisition_multiplier) + logistics_penalty + self.buffer
        projected_profit = (resale_value * (1 - self.platform_fee)) - total_acquisition_cost
        
        # Prevent division by zero if the item is somehow free and has zero logistics cost
        if total_acquisition_cost > 0:
            current_roi = projected_profit / total_acquisition_cost
        else:
            current_roi = 0.0
        
        # The lead is only viable if it hits margin AND liquidity requirements
        is_viable = (current_bid <= max_bid) and (dts <= max_dts) and (max_bid > 0)
        
        return {
            "max_bid": max_bid,
            "projected_profit": round(projected_profit, 2),
            "projected_roi_percentage": round(current_roi * 100, 1),
            "is_viable": is_viable,
            "status": "🟢 GOLD MINE" if is_viable else "🔴 PASS"
        }

# --- Local Test Execution ---
if __name__ == "__main__":
    engine = FinancialEngine(target_roi=5.0) # 500% Target
    
    # Simulating a Gameboy Color found in Pass 1
    # Logistics penalty is $15 (Tier: EASY)
    print("Testing Lead: Nintendo Gameboy Color")
    eval_result = engine.evaluate_lead(
        resale_value=95.00, 
        current_bid=5.00, 
        logistics_penalty=15.0, 
        dts=12.5
    )
    
    for key, value in eval_result.items():
        print(f"{key}: {value}")