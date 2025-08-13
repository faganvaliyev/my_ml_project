import joblib
import pandas as pd
from src.features.build_features import create_preprocessor

def predict(input_data_path, model_path='models/model.pkl', preprocessor_path='models/preprocessor.pkl'):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    df = pd.read_parquet(input_data_path)
    X = preprocessor.transform(df)
    predictions = model.predict(X)
    return predictions

if __name__ == "__main__":
    preds = predict('data/external/new_data.parquet')
    print("Predictions:", preds)
