"""commission_calc.py

Tuned commission rules for Stallion Motors (configurable):
- Per-sale components:
    * Base commission: percentage of Profit OR percentage of Sale Price (configurable).
    * Flat per-sale spiff for special promotions (e.g., $200 for promoting Model X).
    * Returning customer flat bonus per sale.
- Monthly performance components (per salesperson per month):
    * Monthly quota (revenue) -> if exceeded, apply accelerator to base commission for incremental revenue.
    * Tier bonuses: paid as percentage of monthly revenue when crossing thresholds.
    * Bonus for hitting unit target in month (flat bonus).
- Special model spiffs: dictionary mapping Vehicle_Model -> flat spiff per unit sold.
- The module exposes:
    * compute_sale_commission(row)           -> sale-level commission (base + spiffs + returning bonus)
    * compute_leaderboard(df, month=None)   -> monthly leaderboard with full commission breakdown
    * recommended_config                     -> default config dict (for tuning)
"""

from dataclasses import dataclass, field
import pandas as pd
from typing import Dict

# Default configuration (tune these values to match a real compensation plan)
DEFAULT_CONFIG = {
    # Sale-level
    "base_commission_rate_on_profit": 0.04,   # 4% of Profit (preferred)
    "base_commission_rate_on_price": None,    # if set, uses % of Price instead of Profit
    "returning_customer_bonus": 75.0,         # flat bonus per returning-customer sale
    "per_sale_flat_spiff": 0.0,               # general flat spiff per sale
    # Monthly quotas & accelerators
    "monthly_quota_revenue": 80000.0,         # revenue quota per salesperson per month
    "quota_accelerator_rate": 0.015,          # extra % (1.5%) applied to incremental revenue above quota
    "tier_bonuses": [                         # list of (threshold,revenue_pct_bonus)
        (200000.0, 0.025),                    # 2.5% on revenue if >= 200k
        (150000.0, 0.02),                     # 2.0% if >= 150k
        (100000.0, 0.012),                    # 1.2% if >= 100k
        (50000.0, 0.006),                     # 0.6% if >= 50k
    ],
    "unit_target_bonus": {                    # bonus for selling X units in a month
        "units_target": 8,
        "units_target_bonus_amount": 500.0
    },
    # Special model spiffs (flat per unit)
    "model_spiffs": {
        # Example: "Toyota RAV4": 200.0,
    }
}

def recommended_config():
    return DEFAULT_CONFIG.copy()

def compute_sale_commission(row, config=None):
    """Compute commission for a single sale row.
    Expects row with columns: Profit, Price, Vehicle_Model, Customer_Type
    Returns base_commission, spiff, returning_bonus, total_sale_commission
    """
    cfg = DEFAULT_CONFIG if config is None else config
    profit = float(row.get('Profit', 0.0) or 0.0)
    price = float(row.get('Price', 0.0) or 0.0)
    model = row.get('Vehicle_Model', None)
    customer_type = row.get('Customer_Type', 'New')

    # Base commission (prefer profit-based, fallback to price-based)
    if cfg.get('base_commission_rate_on_price'):
        base = cfg['base_commission_rate_on_price'] * price
    else:
        base = cfg['base_commission_rate_on_profit'] * profit

    # Model spiff
    spiff = float(cfg.get('model_spiffs', {}).get(model, 0.0))
    # per-sale flat spiff
    spiff += cfg.get('per_sale_flat_spiff', 0.0)
    # returning customer bonus
    returning = float(cfg.get('returning_customer_bonus', 0.0)) if customer_type == 'Returning' else 0.0

    total = base + spiff + returning
    return {
        'Base_Commission': round(base,2),
        'Spiff': round(spiff,2),
        'Returning_Bonus': round(returning,2),
        'Total_Sale_Level': round(total,2)
    }

def _apply_tier_bonus(revenue, cfg):
    """Return tier bonus amount (flat $) based on revenue tiers in cfg."""
    for threshold, pct in cfg.get('tier_bonuses', []):
        if revenue >= threshold:
            return revenue * pct
    return 0.0

def compute_leaderboard(df, target_month=None, config=None):
    """Compute salesperson monthly leaderboard and commission summary.
    - target_month: period-like (e.g., '2025-10') or None to compute for all months.
    Returns DataFrame with columns:
    ['Salesperson','Month','Revenue','Units','Profit','Sale_Base_Com','Sale_Spiffs','Sale_ReturningBonuses',
     'Base_Commission','Tier_Bonus','Quota_Accelerator','Unit_Target_Bonus','Total_Commission']
    """
    cfg = DEFAULT_CONFIG if config is None else config
    dfc = df.copy()
    dfc['Date'] = pd.to_datetime(dfc['Date'])
    dfc['Month'] = dfc['Date'].dt.to_period('M')

    # Sale-level commissions
    sale_comps = dfc.apply(lambda r: pd.Series(compute_sale_commission(r, cfg)), axis=1)
    dfc = pd.concat([dfc.reset_index(drop=True), sale_comps.reset_index(drop=True)], axis=1)

    # Aggregate monthly per salesperson
    agg = dfc.groupby(['Salesperson','Month']).agg(
        Revenue = ('Price','sum'),
        Units = ('Quantity','sum'),
        Profit = ('Profit','sum'),
        Sale_Base_Com = ('Base_Commission','sum'),
        Sale_Spiffs = ('Spiff','sum'),
        Sale_ReturningBonuses = ('Returning_Bonus','sum')
    ).reset_index()

    # Tier bonuses based on monthly revenue thresholds
    agg['Tier_Bonus'] = agg['Revenue'].apply(lambda r: round(_apply_tier_bonus(r, cfg),2))
    # Quota accelerator for revenue above monthly quota
    quota = cfg.get('monthly_quota_revenue', 0.0)
    acc_rate = cfg.get('quota_accelerator_rate', 0.0)
    def _quota_acc(r):
        if r > quota:
            incremental = r - quota
            return round(incremental * acc_rate,2)
        return 0.0
    agg['Quota_Accelerator'] = agg['Revenue'].apply(_quota_acc)
    # Unit target bonus
    units_t = cfg.get('unit_target_bonus', {}).get('units_target', 999999)
    units_bonus_amt = cfg.get('unit_target_bonus', {}).get('units_target_bonus_amount', 0.0)
    agg['Unit_Target_Bonus'] = agg['Units'].apply(lambda u: units_bonus_amt if u >= units_t else 0.0)

    # Total commission calculation
    agg['Base_Commission'] = agg['Sale_Base_Com'] + agg['Sale_Spiffs'] + agg['Sale_ReturningBonuses']
    agg['Total_Commission'] = (agg['Base_Commission'] + agg['Tier_Bonus'] + agg['Quota_Accelerator'] + agg['Unit_Target_Bonus']).round(2)

    # Optionally filter by month
    if target_month is not None:
        # Accept strings or Period objects
        if isinstance(target_month, str):
            target_month = pd.Period(target_month)
        agg = agg[agg['Month'] == target_month]

    # Sort by Revenue descending
    agg = agg.sort_values(['Month','Revenue'], ascending=[False,False])
    return agg

if __name__ == '__main__':
    # quick test / demo when run directly
    import pandas as pd, json
    sample = pd.DataFrame([{
        'Sale_ID':'S1','Date':'2025-10-05','Vehicle_Model':'Toyota RAV4','Price':50000,'Cost':38000,'Profit':12000,'Quantity':1,'Customer_Type':'New','Salesperson':'Aisha Bello'
    },{
        'Sale_ID':'S2','Date':'2025-10-07','Vehicle_Model':'Toyota RAV4','Price':48000,'Cost':37000,'Profit':11000,'Quantity':1,'Customer_Type':'Returning','Salesperson':'Aisha Bello'
    }])
    print('Config:', json.dumps(DEFAULT_CONFIG, indent=2))
    lb = compute_leaderboard(sample)
    print(lb.head())
