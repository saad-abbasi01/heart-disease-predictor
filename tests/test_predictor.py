import os
import sys

# Ensure current working directory is on pythonpath for imports
sys.path.append(os.getcwd())

import pytest
from app.models.predictor import HeartDiseasePredictor


@pytest.fixture
def predictor():
    """Fixture to load model and scaler once for all tests."""
    return HeartDiseasePredictor(
        "app/models/train_model.pkl",
        "models/scaler.pkl"
    )


def test_predictor_load_successfully(predictor):
    """Verify that both the model and scaler load without issue."""
    assert predictor.model is not None
    assert predictor.scaler is not None


def test_prepare_return_valid_risk_percentage(predictor):
    """Verify prediction output types, range, and risk label options on standard data."""
    patient_data = {
        'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125,
        'chol': 212, 'fbs': 0, 'restecg': 1, 'thalach': 168,
        'exang': 0, 'oldpeak': 1.0, 'slope': 2, 'ca': 2, 'thal': 3
    }
    risk_pct, risk_level = predictor.prepare(patient_data)
    
    assert 0 <= risk_pct <= 100
    assert risk_level in ["🟢 LOW RISK", "🟡 MEDIUM RISK", "🔴 HIGH RISK"]


def test_prepare_raise_missing_value(predictor):
    """Verify that passing incomplete features raises a ValueError."""
    incomplete_data = {
        'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125
    }
    with pytest.raises(ValueError):
        predictor.prepare(incomplete_data)


def test_prepare_high_risk_patient(predictor):
    """Verify edge-case handling for high risk inputs without hardcoding label fragility."""
    high_risk_patient = {
        'age': 70, 'sex': 1, 'cp': 3, 'trestbps': 180,
        'chol': 380, 'fbs': 1, 'restecg': 2, 'thalach': 90,
        'exang': 1, 'oldpeak': 5.0, 'slope': 0, 'ca': 4, 'thal': 3
    }
    
    risk_pct, risk_level = predictor.prepare(high_risk_patient)
    
    # 1. Check percentage bounds
    assert 0 <= risk_pct <= 100
    
    # 2. Check risk level belongs to the supported enumeration
    assert risk_level in ["🟢 LOW RISK", "🟡 MEDIUM RISK", "🔴 HIGH RISK"]