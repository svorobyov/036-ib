import pandas as pd
from lightgbm import (LGBMRegressor, plot_importance, plot_tree)
from IPython.display import display, Markdown  # to display markdown cells
import matplotlib.pyplot as plt
import streamlit as st


CSV_FILE = 'data-6-.csv'
FUTURE_DAYS = 30  # we will forecast for this number of days
FIG_SIZE = (30, 16)  # width x height on my big screen
LAGS = [1, 7, 14, 28]


def read_csv_prophet_df(df):
  """
  Reads data from CSV and prepares three separate DataFrames for locations A, B, and C.

  The function performs the following steps:
  1. Loads the CSV and converts dates.
  2. Trims leading and trailing NaNs for each location.
  3. Interpolates internal missing values.
  4. Resets column names to 'ds' and 'y' as required by Prophet.

  Returns:
      tuple: (dfA, dfB, dfC) prepared DataFrames.
  """
  print(f"NaN in location_A/B/C:\n{df[['location_A', 'location_B', 'location_C']].isna().sum()}")
  df['date'] = pd.to_datetime(df['date'])
  print(f"First date: {df['date'].min()}, last date: {df['date'].max()}")
  dfA = df[['date', f'location_A']]
  dfA.columns = ['ds', 'y']  # Prophet needs these exact two columns
  dfB = df[['date', f'location_B']]
  dfB.columns = ['ds', 'y']  # Prophet needs these exact two columns
  dfC = df[['date', f'location_C']]
  dfC.columns = ['ds', 'y']  # Prophet needs these exact two columns
  # 1. Trim edge NaNs
  print('Range for A:')
  print(dfA['y'].first_valid_index(), dfA['y'].last_valid_index())
  dfA = dfA.loc[dfA['y'].first_valid_index(): dfA['y'].last_valid_index()]
  print('Range for B:')
  print(dfB['y'].first_valid_index(), dfB['y'].last_valid_index())
  dfB = dfB.loc[dfB['y'].first_valid_index(): dfB['y'].last_valid_index()]
  print('Range for C:')
  print(dfC['y'].first_valid_index(), dfC['y'].last_valid_index())
  dfC = dfC.loc[dfC['y'].first_valid_index(): dfC['y'].last_valid_index()]
  # 2. Interpolate internal NaNs
  dfA['y'] = dfA['y'].interpolate()
  dfB['y'] = dfB['y'].interpolate()
  dfC['y'] = dfC['y'].interpolate()
  return dfA, dfB, dfC  # Has/shows: Pandas(Index=1337, ds=Timestamp('2025-08-31 00:00:00'), y=10042), Index is Pandas metadata, ignored by Prophet


def lightgbm_experiment(df, location='A'):
  """
  Fits a LightGBM model to the data and visualizes feature importance and trees.

  Args:
      df (pd.DataFrame): Input dataframe with 'ds' and 'y'.
      location (str): Name of the location for titles.
  """
  # Simple feature engineering
  df = df.copy()
  df['dow'] = df['ds'].dt.dayofweek
  df['month'] = df['ds'].dt.month

  features = ['dow', 'month']
  X = df[features]
  y = df['y']

  model = LGBMRegressor(n_estimators=100)
  model.fit(X, y)

  # N_ROWS = 100
  df = df.sort_values('ds')
  # print(df.head(N_ROWS))

  # Compute lags: see the explanation below this function
  for lag in LAGS:
    df[f'lag_{lag}'] = df['y'].shift(lag)

  # print(df.head(40))

  # Rolling stats
  df['rolling_mean_7'] = df['y'].shift(1).rolling(7).mean()
  df['rolling_std_7'] = df['y'].shift(1).rolling(7).std()

  # Calendar features
  df['dow'] = df['ds'].dt.dayofweek
  df['month'] = df['ds'].dt.month
  df['is_weekend'] = (df['dow'] >= 5).astype(int)

  df = df.dropna()
  # print(df.head(N_ROWS))

  features = [col for col in df.columns if col not in ['ds', 'y']]
  # print(features)

  # Turn it in tabular data: X->y, features->y,
  # ['lag_1', 'lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_std_7', 'dow', 'month', 'is_weekend'] -> y
  # I.e., current day depends on day-1, -7, -14, -28, ...
  X = df[features]
  y = df['y']

  model = LGBMRegressor(
    objective='poisson',  # important for count data
    n_estimators=500,
    learning_rate=0.05
  )

  model.fit(X, y)

  # Predictions vs actuals; see the explanation below.
  y_pred = model.predict(X)

  fig, ax = plt.subplots(figsize=FIG_SIZE)
  ax.plot(df['ds'], y, label='actual')
  ax.plot(df['ds'], y_pred, label='predicted')
  ax.legend()
  st.pyplot(fig)

  # 1. Plot Importance with increased scale
  display(Markdown(f"### Location {location}: LightGBM Feature Importance"))
  fig, ax = plt.subplots(figsize=FIG_SIZE)
  plot_importance(model, ax=ax)
  plt.title(f'Feature Importance - {location}')
  # plt.show()
  ax.legend()
  st.pyplot(fig)


  # 2. Plot Tree with increased scale
  display(Markdown(f"### Location {location}: LightGBM Decision Tree (index 0)"))
  # dpi=100 or 200 make the lines and text much sharper
  fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=150)
  plot_tree(model, tree_index=0, ax=ax)
  plt.title(f'Decision Tree - {location}')
  # plt.show()
  ax.legend()
  st.pyplot(fig)

  residuals = y - y_pred

  display(Markdown(f"### Location {location}: LightGBM residuals (y-y_pred)"))
  fig, ax = plt.subplots(figsize=FIG_SIZE)
  ax.plot(df['ds'], residuals)
  ax.axhline(0)
  # plt.show()
  ax.legend()
  st.pyplot(fig)

  # Now we proceed with real predictions for future deliveries, the number of which we do NOT know
  # A recursive/iterative prediction loop is needed because the features (lags, rolling stats) depend on previous predictions once we move beyond the last observed point

  history = df.copy()  # will be appended to

  horizon = FUTURE_DAYS
  future_rows = []

  for i in range(horizon):
    next_date = history['ds'].max() + pd.Timedelta(days=1)

    row = {'ds': next_date}

    # --- lag features ---
    for lag in LAGS:
      row[f'lag_{lag}'] = history['y'].iloc[-lag]

    # --- rolling features ---
    row['rolling_mean_7'] = history['y'].iloc[-7:].mean()
    row['rolling_std_7'] = history['y'].iloc[-7:].std()

    # --- calendar features ---
    row['dow'] = next_date.dayofweek
    row['month'] = next_date.month
    row['is_weekend'] = int(row['dow'] >= 5)

    # convert to DataFrame
    X_next = pd.DataFrame([row]).drop(columns=['ds'])

    # predict
    y_pred = model.predict(X_next)[0]

    row['y'] = y_pred

    # append
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    future_rows.append(row)

  forecast_df = pd.DataFrame(future_rows)

  # Plot actual + forecast
  display(Markdown(f"### Location {location}: LightGBM history and forecast"))
  fig, ax = plt.subplots(figsize=FIG_SIZE)

  # historical
  ax.plot(df['ds'], df['y'], label='actual')

  # forecast
  ax.plot(forecast_df['ds'], forecast_df['y'], label='forecast')

  ax.legend()
  st.pyplot(fig)

  # Zooming
  tail = 60  # last 60 days
  fig, ax = plt.subplots(figsize=FIG_SIZE)
  display(Markdown(f"### Location {location}: LightGBM history and zoomed forecast"))
  ax.plot(df['ds'].iloc[-tail:], df['y'].iloc[-tail:], label='actual')
  ax.plot(forecast_df['ds'], forecast_df['y'], label='forecast')
  ax.legend()
  st.pyplot(fig)


if __name__ == '__main__':
  st.title("LightGBM model for your time series data")
  uploaded_file = st.file_uploader("Load csv file", type=['csv'])
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    dfA, dfB, dfC = read_csv_prophet_df(df)
    lightgbm_experiment(dfA, location='A')
