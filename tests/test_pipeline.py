"""
test_pipeline.py
Basic unit tests for preprocessing and prediction logic.

Run:
    pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
from preprocess import load_raw_data, encode_categoricals, CATEGORICAL_COLS, TARGET_COL


def test_raw_data_loads():
    df = load_raw_data("data/customer_churn.csv")
    assert not df.empty
    assert TARGET_COL in df.columns


def test_no_missing_values():
    df = load_raw_data("data/customer_churn.csv")
    assert df.isnull().sum().sum() == 0


def test_target_is_binary():
    df = load_raw_data("data/customer_churn.csv")
    assert set(df[TARGET_COL].unique()).issubset({0, 1})


def test_encode_categoricals_returns_numeric():
    df = load_raw_data("data/customer_churn.csv")
    df_encoded, encoders = encode_categoricals(df, fit=True)
    for col in CATEGORICAL_COLS:
        assert pd.api.types.is_numeric_dtype(df_encoded[col])
    assert len(encoders) == len(CATEGORICAL_COLS)


def test_encoders_are_consistent():
    df = load_raw_data("data/customer_churn.csv")
    df_encoded, encoders = encode_categoricals(df, fit=True)
    # re-applying the same encoders should give identical results
    df_encoded_2, _ = encode_categoricals(df, encoders=encoders, fit=False)
    pd.testing.assert_series_equal(df_encoded["contract_type"], df_encoded_2["contract_type"])
