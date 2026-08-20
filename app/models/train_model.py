import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , precision_score ,recall_score,f1_score
import joblib
from app.models.preprocessing import DataPreprocessor

def train_model():
    preprocessor=DataPreprocessor()
    X_train,X_test,Y_train,Y_test=preprocessor.full_pipeline("data/raw/heart_disease.csv")
    
    #Now train the model till our limits..
    model=RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    #start trained the model now
    print("Model start fro training")
    model.fit(X_train,Y_train)
    print("Training completed")
    
    #prediction of the model for testing
    y_pred=model.predict(X_test)
    
    #Now process the all scores
    accuracy=accuracy_score(Y_test,y_pred)
    precision=precision_score(Y_test,y_pred)
    recall=recall_score(Y_test,y_pred)
    f1=f1_score(Y_test,y_pred)
    
    #model explanation or output
    print("===Model Explanation===")
    print(f"Accuracy_score:{accuracy:.2f}")
    print(f"Precision_score:{precision:.2f}")
    print(f"Recall_score:{recall:.2f}")
    print(f"F1_score:{f1:.2f}")
    
    #now saing file and scaler file into models
    
    joblib.dump(model,"app/models/train_model.pkl")
    print("\nSave the model in the train_model.pkl")
    
    #now the save scaler
    preprocessor.save_scaler("models/scaler.pkl")
    
    return model ,accuracy

if __name__ =="__main__":
    train_model()
    