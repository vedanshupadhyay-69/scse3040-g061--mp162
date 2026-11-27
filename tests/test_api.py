from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    payload = {
        "patients_ahead": 5,
        "doctors_on_duty": 2,
        "arrival_hour": 10,
        "avg_consult_min": 15,
        "lab_tests_pending": 3,
        "distance_km": 4.5,
        "department": "General",
        "wing": "A"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_wait_min" in response.json()


def test_negative_patients_ahead():
    payload = {
        "patients_ahead": -1,
        "doctors_on_duty": 2,
        "arrival_hour": 10,
        "avg_consult_min": 15,
        "lab_tests_pending": 3,
        "distance_km": 4.5,
        "department": "General",
        "wing": "A"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_zero_doctors():
    payload = {
        "patients_ahead": 5,
        "doctors_on_duty": 0,
        "arrival_hour": 10,
        "avg_consult_min": 15,
        "lab_tests_pending": 3,
        "distance_km": 4.5,
        "department": "General",
        "wing": "A"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_arrival_hour():
    payload = {
        "patients_ahead": 5,
        "doctors_on_duty": 2,
        "arrival_hour": 25,
        "avg_consult_min": 15,
        "lab_tests_pending": 3,
        "distance_km": 4.5,
        "department": "General",
        "wing": "A"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_field():
    payload = {
        "patients_ahead": 5,
        "doctors_on_duty": 2,
        "arrival_hour": 10,
        "avg_consult_min": 15,
        "lab_tests_pending": 3,
        "distance_km": 4.5,
        "department": "General"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422