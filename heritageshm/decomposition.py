"""
Module: decomposition.py
NeuralProphet grey-box wrapper for structural decomposition.
Handles model configuration, training, component extraction, and residual computation.
"""

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, set_log_level


def build_neuralprophet_df(df, target_col, regressor_cols):
    """
    Converts the internal DataFrame to the NeuralProphet 'ds/y' format,
    appending all exogenous regressors as additional columns.

    Parameters:
    - df: DataFrame with DatetimeIndex and target + regressor columns.
    - target_col: Name of the structural target column (e.g., 'absinc').
    - regressor_cols: List of exogenous regressor column names.

    Returns:
    - df_np: NeuralProphet-ready DataFrame with columns ['ds', 'y', *regressors].
    """
    df_np = df[[target_col] + regressor_cols].copy()
    df_np = df_np.reset_index()
    df_np.columns = ['ds'] + ['y'] + regressor_cols
    df_np['ds'] = pd.to_datetime(df_np['ds'])
    return df_np


def configure_model(regressor_cols,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    n_lags=24,
                    ar_reg=0.1,
                    trend_reg=0.5,
                    seasonality_reg=0.1,
                    epochs=100,
                    learning_rate=0.001,
                    batch_size=64):
    """
    Instantiates and configures a NeuralProphet model for grey-box SHM decomposition.

    Design rationale:
    - Trend captures slow irreversible structural drift.
    - Yearly + weekly Fourier seasonalities capture thermally driven periodic response.
    - Daily seasonality is disabled by default (dominated by ERA5-Land hourly proxy).
    - AR lags (n_lags) capture short-term structural memory not explained by the proxy.
    - Lagged environmental proxies are added as future regressors (known at all times
      because they come from reanalysis data with no gaps).
    - Regularisation on trend, seasonality, and AR prevents overfitting to imputed data.

    Parameters:
    - regressor_cols: List of exogenous regressor column names.
    - yearly_seasonality: Include Fourier yearly seasonality (default True).
    - weekly_seasonality: Include Fourier weekly seasonality (default True).
    - daily_seasonality: Include Fourier daily seasonality (default False).
    - n_lags: Number of autoregressive lags (default 24 h).
    - ar_reg: AR regularisation strength (default 0.1).
    - trend_reg: Trend regularisation strength (default 0.5).
    - seasonality_reg: Seasonality regularisation strength (default 0.1).
    - epochs: Training epochs (default 100).
    - learning_rate: Adam learning rate (default 0.001).
    - batch_size: Mini-batch size (default 64).

    Returns:
    - model: Configured NeuralProphet instance (unfitted).
    """
    set_log_level("ERROR")

    model = NeuralProphet(
        trend_reg=trend_reg,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_reg=seasonality_reg,
        n_lags=n_lags,
        ar_reg=ar_reg,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        collect_metrics=True,
    )

    for col in regressor_cols:
        model.add_future_regressor(col, regularization=0.05)

    return model


def train_model(model, df_np, valid_fraction=0.15):
    """
    Trains the NeuralProphet model and returns the fitted model and metrics.

    Parameters:
    - model: Configured NeuralProphet instance.
    - df_np: NeuralProphet-format DataFrame ('ds', 'y', regressors).
    - valid_fraction: Fraction of data reserved for validation (default 0.15).

    Returns:
    - model: Fitted NeuralProphet instance.
    - metrics: DataFrame of training/validation loss per epoch.
    """
    df_train, df_val = model.split_df(df_np, valid_p=valid_fraction)
    metrics = model.fit(df_train, validation_df=df_val, progress="bar")
    return model, metrics, df_train, df_val


def extract_components(model, df_np):
    """
    Generates NeuralProphet predictions and extracts decomposed components.

    NeuralProphet's predict() returns one row per forecast step with columns:
      - yhat1         : total fitted value (1-step-ahead)
      - trend         : trend component
      - season_yearly : yearly Fourier seasonality
      - season_weekly : weekly Fourier seasonality
      - ar1           : autoregressive contribution (lag 1)
      - future_regressor_<name> : exogenous regressor contribution

    This function assembles a clean component DataFrame indexed by datetime.

    Parameters:
    - model: Fitted NeuralProphet instance.
    - df_np: NeuralProphet-format DataFrame ('ds', 'y', regressors).

    Returns:
    - df_pred: DataFrame with 'ds', 'y', 'yhat1', and all component columns.
    - component_cols: List of identified component column names.
    """
    df_pred = model.predict(df_np)
    df_pred = df_pred.set_index('ds')
    df_pred.index = pd.to_datetime(df_pred.index)

    # Identify component columns dynamically
    exclude = {'y', 'yhat1'}
    component_cols = [c for c in df_pred.columns if c not in exclude]

    return df_pred, component_cols


def compute_residuals(df_pred, target_col='y', yhat_col='yhat1'):
    """
    Computes structural residuals as the difference between observed and fitted values.

    e_t = y_t - yhat1_t

    These residuals represent the structural signal not explained by trend,
    seasonality, AR memory, or environmental proxies. They form the input
    to the EWMA/CUSUM control charts in monitoring.py.

    Parameters:
    - df_pred: Output DataFrame from extract_components().
    - target_col: Observed target column name (default 'y').
    - yhat_col: Fitted value column name (default 'yhat1').

    Returns:
    - residuals: pd.Series of structural residuals indexed by datetime.
    """
    residuals = df_pred[target_col] - df_pred[yhat_col]
    residuals.name = 'residual'
    return residuals.dropna()


def summarise_components(df_pred, component_cols):
    """
    Prints a variance attribution table: how much each component contributes
    to the total explained variance of the fitted signal.

    Parameters:
    - df_pred: Output of extract_components().
    - component_cols: List of component column names.
    """
    records = []
    total_var = df_pred['yhat1'].var()
    for col in component_cols:
        if col in df_pred.columns:
            var = df_pred[col].var()
            pct = 100 * var / total_var if total_var > 0 else 0.0
            records.append({'Component': col, 'Variance': round(var, 4), '% of Total': round(pct, 2)})

    summary = pd.DataFrame(records).sort_values('% of Total', ascending=False)
    print("\n--- Component Variance Attribution ---")
    print(summary.to_string(index=False))
    return summary