import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

def tune_xgb(X_train, y_train, preprocessor):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBClassifier(**params, random_state=42, eval_metric='logloss'))
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=90)
        return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params