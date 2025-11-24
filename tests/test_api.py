import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pytest
from flask import Flask
from app import app
from train import spam_detector


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200


def test_api_integration_with_model(monkeypatch):
    """
    Teste real → chama spam_detector de verdade.
    """

    def fake_spam_detector(text):
        return {"label": 1, "label_name": "spam", "probabilities": {0: 0.1, 1: 0.9}}

    monkeypatch.setattr("train.spam_detector", fake_spam_detector)

    result = spam_detector("Congratulations, you won!")
    assert result["label"] == 1
    assert result["label_name"] == "spam"


def test_api_full_flow(client, monkeypatch):
    """
    Teste de integração API + modelo (mockado).
    """

    def fake_spam_detector(text):
        return {"label": 0, "label_name": "ham"}

    monkeypatch.setattr("train.spam_detector", fake_spam_detector)

    response = client.post(
        "/predict",
        json={"text": "hello friend"},
    )

    assert response.status_code == 200
    data = response.json

    assert data["label_name"] == "ham"