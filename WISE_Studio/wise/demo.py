"""Built-in demo event log (generic order-to-delivery), with planted hotspots,
so the app is try-able with zero setup."""
import numpy as np
import pandas as pd


def generate_demo_log(n_cases=3000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = ["North", "South", "East", "West"]
    products = ["Standard", "Custom", "Bulk", "Express"]
    teams = [f"agent_{i:02d}" for i in range(12)]
    rows = []
    t0 = pd.Timestamp("2024-01-05")
    for i in range(n_cases):
        cid = f"ORD-{i:05d}"
        region = rng.choice(regions, p=[0.4, 0.3, 0.2, 0.1])
        product = rng.choice(products, p=[0.5, 0.2, 0.2, 0.1])
        start = t0 + pd.Timedelta(days=float(rng.uniform(0, 330)))
        t = start
        amount = float(rng.lognormal(6, 1))

        def ev(act, dt_days, res=None):
            nonlocal t
            t = t + pd.Timedelta(days=float(dt_days))
            rows.append({"order_id": cid, "step": act, "when": t,
                         "handler": res or rng.choice(teams),
                         "region": region, "product_type": product,
                         "amount_eur": round(amount, 2)})

        ev("Create Order", 0)
        ev("Approve Order", rng.exponential(0.6))
        # planted hotspot 1: South×Custom — repeated picking (rework mechanism)
        n_pick = 1 + (rng.poisson(2.2) if (region == "South" and product == "Custom")
                      else rng.poisson(0.15))
        for _ in range(max(n_pick, 1)):
            ev("Pick Items", rng.exponential(0.4))
        ev("Ship Order", rng.exponential(0.8))
        # planted hotspot 2: North×Bulk — slow invoicing (ageing reservoir)
        inv_lag = rng.exponential(9.0) if (region == "North" and product == "Bulk") \
            else rng.exponential(2.0)
        ev("Send Invoice", inv_lag)
        if rng.random() < (0.18 if region == "East" else 0.03):
            ev("Cancel Invoice", rng.exponential(1.0))
            ev("Send Invoice", rng.exponential(1.0))
        if rng.random() > 0.08:  # 8% never paid (open)
            ev("Receive Payment", rng.exponential(6.0))
    return pd.DataFrame(rows)
