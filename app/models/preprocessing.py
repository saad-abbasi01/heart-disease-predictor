import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class DataPreprocessor:
    def __init__(self):
        
        self.scaler=StandardScaler()
        self.feature_columns=[
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
        self.target_column="target"
        
    def load_data(self,filepath):
        df=pd.read_csv(filepath)
        return df

    def validate_data(self,df):
        missing=df.isnull().sum().sum()
        
        if missing > 0:
            print(f"Missing {missing} value in this data..")
        else:
            print(f"Missing zero value in this data...")
        return df  
    def split_feature_target(self,df):
        """Split feature from the target as questions form the answers"""
        X=df[self.feature_columns]
        Y=df[self.target_column]
        
        return X,Y
    def split_feature_target_data(self,X,Y,test_size=0.30,random_state=42):
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=random_state,stratify=Y)
        return X_train,X_test,Y_train,Y_test

    def scale_feature(self,X_train,X_test):
        X_train_scaled=self.scaler.fit_transform(X_train)
        X_test_scaled=self.scaler.transform(X_test)
        return X_train_scaled ,X_test_scaled
    def save_scaler(self,filepath):
        joblib.dump(self.scaler,filepath)
        print(f"Save scaler to filepath{filepath}")
        
    def load_scaler(self,filepath):
        self.scaler=joblib.load(filepath)
        
    def full_pipeline(self,filepath):
        df=self.load_data(filepath)
        print(f"loaded data{len(df)} patient records")
        
        df=self.validate_data(df)
        X,Y=self.split_feature_target(df)
        
        X_train,X_test,Y_train,Y_test=self.split_feature_target_data(X,Y)
        print(f"Traing set:{len(X_train)} patients.")
        print(f"Testing set:{len(X_test)} patients.")
        
        X_train_scaled,X_test_scaled=self.scale_feature(X_train,X_test)
        return X_train_scaled, X_test_scaled ,Y_train ,Y_test
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline(
        "data/raw/heart_disease.csv"
    )
    print("\nPreprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")