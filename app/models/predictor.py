import numpy as np
from pathlib import Path
import joblib
import shap
class HeartDiseasePredictor:
    
    def __init__(self,model_path,scaler_path):
        self.model=joblib.load(model_path)
        self.scaler=joblib.load(scaler_path)
        self.feature_cols=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
    def prepare(self,patient_data:dict):
        features=np.array([patient_data[cols] for cols in self.feature_cols]).reshape(1,-1)
        
        feature_scaled=self.scaler.transform(features)
        risk_prob=self.model.predict_proba(feature_scaled)[0][1]
        risk_percentage=int(risk_prob * 100)
        
        if risk_percentage < 30:
            risk_level = "🟢 LOW RISK"
        elif risk_percentage < 70:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🔴 HIGH RISK"

        return risk_percentage, risk_level
    

    def explain(self,patient_data:dict):
        
        features=np.array([patient_data[col] for col in self.feature_cols]).reshape(1,-1)
        feature_scaled=self.scaler.transform(features)
        explainer=shap.TreeExplainer(self.model)
        shape_values=explainer.shap_values(feature_scaled)
        
        contributers=shape_values[0][:, 1]
        
        return list(zip(self.feature_cols,contributers))
    
if __name__ == "__main__":
    predictor=HeartDiseasePredictor(
        "app/models/train_model.pkl",
        "models/scaler.pkl"
    )
    test_patient = {
        'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125,
        'chol': 212, 'fbs': 0, 'restecg': 1, 'thalach': 168,
        'exang': 0, 'oldpeak': 1.0, 'slope': 2, 'ca': 2, 'thal': 3
    }
            
    risk_per,risk_level=predictor.prepare(test_patient)
    print(f"Risk percentage is :{risk_per:.2f} and Risk level is:{risk_level}")
        