from churn.data import generate_customers
from churn.eda import summarize
from churn.features import CATEGORICAL, NUMERIC, TARGET
from churn.model import train_and_evaluate


def test_generate_shape_and_columns():
    df = generate_customers(500, seed=1)
    assert len(df) == 500
    for col in NUMERIC + CATEGORICAL + [TARGET]:
        assert col in df.columns
    assert set(df[TARGET].unique()) <= {0, 1}


def test_generation_is_reproducible():
    a = generate_customers(200, seed=7)
    b = generate_customers(200, seed=7)
    assert a.equals(b)


def test_month_to_month_churns_more():
    s = summarize(generate_customers(3000, seed=3))
    by = s["churn_by_contract"]
    assert by["Month-to-month"] > by["Two year"]


def test_model_beats_random(tmp_path):
    df = generate_customers(1500, seed=5)
    res = train_and_evaluate(df, tmp_path / "figs", tmp_path / "model.joblib")
    assert res["test"]["roc_auc"] > 0.7
    assert (tmp_path / "model.joblib").exists()
