from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders.cat_boost import CatBoostEncoder

def create_preprocessor(numeric_features, categoric_features):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])
    categoric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', CatBoostEncoder())
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categoric_transformer, categoric_features)
    ])
