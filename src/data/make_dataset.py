import pandas as pd
import numpy as np

def load_and_clean_data(path):
    df = pd.read_parquet(path)
    df_subset = df.iloc[:, list(range(1, 14)) + [-1]]
    df_subset = df_subset.sample(frac=1, random_state=41).reset_index(drop=True)
    
    binary_cols = ['is_dualsim', 'is_featurephone', 'is_smartphone', 'target']
    to_float_cols = ['age', 'age_dev', 'dev_num']
    for col in binary_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].astype(int)
    for col in to_float_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].astype('float64')
    
    df_subset['smart_dual'] = df_subset['is_smartphone'] * df_subset['is_dualsim']
    df_subset = df_subset.replace(['None', None], np.nan)
    df_subset['region'] = df_subset['region'].replace('GENJE, GENJE', 'GENJE')
    df_subset.loc[df_subset['age'] == 1941.0, 'age'] = 2025 - 1941
    
    return df_subset
