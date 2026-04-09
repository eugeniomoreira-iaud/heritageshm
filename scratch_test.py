import pandas as pd
import numpy as np

def characterize_gaps(df, target_col, max_impute_gap=0):
    df_imputed = df.copy()
    if max_impute_gap > 0:
        df_imputed[target_col] = df_imputed[target_col].interpolate(method='linear', limit=max_impute_gap)
        
    missing_mask = df_imputed[target_col].isnull()
    if not missing_mask.any():
        return df_imputed, pd.Series(dtype=float), pd.Series(dtype=int)
        
    gap_blocks = missing_mask.astype(int).groupby(missing_mask.astype(int).diff().ne(0).cumsum())
    gap_lengths = gap_blocks.sum()[gap_blocks.sum() > 0]
    return gap_lengths

df = pd.DataFrame({'absinc': [1, np.nan, np.nan, np.nan, 4, np.nan, np.nan, 8]})
print("Original:")
print(characterize_gaps(df, 'absinc', 0).value_counts())

print("Imputed limit=2:")
print(characterize_gaps(df, 'absinc', 2).value_counts())
