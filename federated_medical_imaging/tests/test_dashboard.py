"""Tests for the Flask dashboard."""
import pytest
import json
import ast
from dashboard.app import create_app

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app, socketio = create_app()
    app.config.update({
        "TESTING": True,
    })
    yield app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_index_route(client):
    """GET / returns 200"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.data or b"<!DOCTYPE html>" in response.data

def test_status_api(client):
    """GET /api/status returns valid JSON with expected keys"""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "training_active" in data
    assert "current_round" in data
    assert "total_rounds" in data
    assert "model_type" in data
    assert "strategy" in data
    assert "elapsed_time" in data
    assert "num_clients" in data

def test_metrics_api(client):
    """GET /api/metrics returns valid JSON"""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "rounds" in data
    assert isinstance(data["rounds"], list)

def test_clients_api(client):
    """GET /api/clients returns valid JSON"""
    response = client.get("/api/clients")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "clients" in data
    assert isinstance(data["clients"], list)

def test_push_metrics(client):
    """POST /api/push-metrics with valid JSON returns 200"""
    payload = {
        "round": 1,
        "global_metrics": {"accuracy": 0.5, "loss": 1.2},
        "client_metrics": {
            "Hospital_A": {"accuracy": 0.45, "loss": 1.3, "samples": 300}
        }
    }
    response = client.post("/api/push-metrics", json=payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert data["round"] == 1

def test_push_metrics_invalid(client):
    """POST with invalid JSON returns 400"""
    payload = {"wrong_key": "no_round"}
    response = client.post("/api/push-metrics", json=payload)
    assert response.status_code == 400

def test_model_info(client):
    """GET /api/model-info returns valid JSON"""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "architecture" in data
    assert "parameters" in data

def test_comparison_api(client):
    """GET /api/comparison returns valid JSON"""
    response = client.get("/api/comparison")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "federated" in data
    assert "centralized" in data
