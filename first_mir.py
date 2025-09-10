import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

train_set = pd.read_csv("waiting_times_train.csv")
y_train = train_set.iloc[:, -1]

weather = pd.read_csv("weather_data.csv")
merged_train = train_set.merge(weather, on="DATETIME", how="left")
merged_train = pd.get_dummies(merged_train, columns=["ENTITY_DESCRIPTION_SHORT"])
merged_train["DATETIME"] = pd.to_datetime(merged_train["DATETIME"], format="%Y-%m-%d %H:%M:%S")

start_covid = pd.Timestamp("2020-03-01 00:00:00")
end_covid   = pd.Timestamp("2021-12-31 23:59:59")
merged_train["is_covid"] = merged_train["DATETIME"].between(start_covid, end_covid).astype(int)

merged_train["day_of_week"] = merged_train["DATETIME"].dt.dayofweek
merged_train["is_weekend"] = merged_train["day_of_week"].isin([5,6]).astype(int)
merged_train = pd.get_dummies(merged_train, columns=["day_of_week"])
merged_train["month"] = merged_train["DATETIME"].dt.month
merged_train["is_vacation"] = merged_train["month"].isin([6,7,8]).astype(int)
merged_train["month_sin"] = np.sin(2 * np.pi * merged_train["month"] / 12)
merged_train["month_cos"] = np.cos(2 * np.pi * merged_train["month"] / 12)
merged_train["hour"] = merged_train["DATETIME"].dt.hour
merged_train["hour_sin"] = np.sin(2 * np.pi * merged_train["hour"] / 24)
merged_train["hour_cos"] = np.cos(2 * np.pi * merged_train["hour"] / 24)
merged_train = merged_train.drop(columns=["DATETIME", "hour", "month"])
merged_train = merged_train.drop(columns=["WAIT_TIME_IN_2H"])
"""
merged_train["temp_diff"] = merged_train["feels_like"] - merged_train["temp"]
merged_train["temp_humidity"] = merged_train["temp"] * merged_train["humidity"]/100
merged_train["wind_effect"] = merged_train["wind_speed"] * (1 + merged_train["clouds_all"]/100)

merged_train["precipitation"] = merged_train["rain_1h"].fillna(0) + merged_train["snow_1h"].fillna(0)
merged_train['wait_rolling_mean_3h'] = merged_train['CURRENT_WAIT_TIME'].rolling(3).mean().fillna(merged_train['CURRENT_WAIT_TIME'].median())
merged_train["next_parade_time"] = merged_train[["TIME_TO_PARADE_1","TIME_TO_PARADE_2"]].min(axis=1)
merged_train["parade_in_progress"] = ((merged_train["TIME_TO_PARADE_1"] <= 0) | (merged_train["TIME_TO_PARADE_2"] <= 0)).astype(int)
merged_train["night_show_in_progress"] = (merged_train["TIME_TO_NIGHT_SHOW"] <= 0).astype(int)
merged_train["adjusted_load"] = merged_train["CURRENT_WAIT_TIME"] / (merged_train["ADJUST_CAPACITY"] + 1)
merged_train["total_event_wait"] = merged_train[["TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"]].sum(axis=1)
"""
"""
cols_train_drop = ["dew_point", "pressure", "day_of_week_0", "day_of_week_1", "day_of_week_2", "day_of_week_3", "day_of_week_4", "day_of_week_5", "day_of_week_6"]#, "rain_1h", "snow_1h"]
merged_train = merged_train.drop(columns=[c for c in cols_train_drop if c in merged_train.columns])
"""





# Features
features = merged_train.drop(columns=["CURRENT_WAIT_TIME"])  # ou ta colonne cible
target = merged_train["CURRENT_WAIT_TIME"]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    features, target, test_size=0.2, random_state=42
)
