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
    response = client.get("/home")
    assert response.status_code == 200


def test_api_integration_with_model(monkeypatch):

    def fake_spam_detector(text):

        return {"label": 1, "probabilities": {0: 0.1, 1: 0.9}}

    monkeypatch.setattr("train.spam_detector", fake_spam_detector)

    result = spam_detector("Your email was randomly selected as the grand prize winner in our monthly loyalty program. To claim your $5,000 Amazon Gift Card, simply confirm your identity at the secure link below:")
    assert result["label"] == 1



def test_api_full_flow(client, monkeypatch):

    def fake_spam_detector(text):

        return {"label": 0}

    monkeypatch.setattr("train.spam_detector", fake_spam_detector)

    response = client.post(
        "/predict",
        json={"text": "Subject: request for a quote for corporate website development. Hello, I would like to know the price."},
    )

    assert response.status_code == 200
    data = response.json

    assert data["label"] == 0