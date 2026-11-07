"""Run the full churn pipeline: generate data -> EDA -> train -> report."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from churn.data import generate_customers  # noqa: E402
from churn.eda import make_figures, summarize  # noqa: E402
from churn.model import train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000, help="number of customers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_customers(args.rows, args.seed)
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "customers.csv", index=False)

    fig_dir = ROOT / "reports" / "figures"
    summary = summarize(df)
    make_figures(df, fig_dir)
    results = train_and_evaluate(df, fig_dir, ROOT / "models" / "churn_model.joblib")

    report = {"data_summary": summary, "modeling": results}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
