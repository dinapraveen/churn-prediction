"""Generate a realistic synthetic telecom customer dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
INTERNET = ["Fiber optic", "DSL", "None"]


def generate_customers(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic customer table with a `churn` label (0/1).

    Churn probability depends on contract type, tenure, monthly charges,
    support calls and payment method, so the data has learnable signal.
    """
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(PAYMENTS, n, p=[0.35, 0.22, 0.22, 0.21])
    tenure = rng.integers(1, 73, n)
    senior = rng.binomial(1, 0.16, n)
    support_calls = rng.poisson(1.5, n)

    base = np.select(
        [internet == "Fiber optic", internet == "DSL"], [80.0, 55.0], default=25.0
    )
    monthly = np.round(base + rng.normal(0, 12, n), 2).clip(18, 120)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)

    logit = (
        -1.2
        + np.select(
            [contract == "Month-to-month", contract == "One year"], [1.4, 0.0], -1.3
        )
        - 0.035 * tenure
        + 0.018 * (monthly - 60)
        + 0.35 * support_calls
        + 0.6 * (payment == "Electronic check")
        + 0.4 * senior
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "senior_citizen": senior,
            "support_calls": support_calls,
            "monthly_charges": monthly,
            "total_charges": total,
            "churn": churn,
        }
    )
    # Inject a few missing values, as real data would have.
    missing = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[missing, "total_charges"] = np.nan
    return df
