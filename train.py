from src.data.make_dataset import load_and_clean_data
from src.features.build_features import create_preprocessor
from src.models.train_model import train_and_evaluate
from src.models.tune_model import tune_xgb
import joblib

if __name__ == "__main__":
    df = load_and_clean_data('data/external/data.parquet')
    X, y = df.drop('target', axis=1), df['target']

    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    categoric_cols = X.select_dtypes(exclude='number').columns.tolist()

    preprocessor = create_preprocessor(numeric_cols, categoric_cols)
    metrics = train_and_evaluate(X, y, preprocessor)
    print("Training metrics:", metrics)

    best_params = tune_xgb(X, y, preprocessor)
    print("Best Parameters:", best_params)

    joblib.dump(best_params, 'models/best_params.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
