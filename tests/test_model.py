import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import pandas as pd
import joblib
import pytest

import train
from train import (
    normalize_text,
    etl_pipeline,
    preprocessing_and_train,
    spam_detector,
    MODEL_PATH,
    VECTORIZER_PATH,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def test_normalize_text_basic():

    texto = "Hello!!! This is a TEST message 123."
    resultado = normalize_text(texto)

    assert isinstance(resultado, str)

    assert "Hello" not in resultado  
    assert "test" in resultado or "messag" in resultado


def test_etl_pipeline_balance():
    df = pd.DataFrame({
        "text": ["spam message", "ham message", "buy now", "hello friend"],
        "label": [1, 0, 1, 0],
    })

    processed = etl_pipeline(df, "text", "label")
    spam_count = sum(processed["label"] == 1)
    ham_count = sum(processed["label"] == 0)

    assert spam_count == ham_count
    assert "text_norm" in processed.columns


def test_preprocessing_and_train(tmp_path, monkeypatch):
    
    csv_file = tmp_path / "emails.csv"
    df = pd.DataFrame({
        "text": ["buy now", "hello friend", "cheap pills", "meeting tomorrow"],
        "label": [1, 0, 1, 0],
    })
    df.to_csv(csv_file, index=False)

    monkeypatch.setattr("train.CSV_PATH", str(csv_file))
    monkeypatch.setattr("train.MODEL_PATH", str(tmp_path / "spam_model.joblib"))
    monkeypatch.setattr("train.VECTORIZER_PATH", str(tmp_path / "vec.joblib"))

    result = preprocessing_and_train(csv_path=str(csv_file))

    assert os.path.exists(train.MODEL_PATH)
    assert os.path.exists(train.VECTORIZER_PATH)
    assert "accuracy" in result
    assert result["accuracy"] >= 0


def test_spam_detector_prediction(tmp_path, monkeypatch):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(["spam message", "hello friend"])
    model = MultinomialNB()
    model.fit(X, [1, 0])

    joblib.dump(model, tmp_path / "model.joblib")
    joblib.dump(vectorizer, tmp_path / "vec.joblib")

    monkeypatch.setattr("train.MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr("train.VECTORIZER_PATH", str(tmp_path / "vec.joblib"))

    result = spam_detector("Hello friend")

    assert "label" in result