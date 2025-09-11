import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# --- Fonction pour marquer les événements spéciaux ---
def mark_special_event(dt):
    fixed_holidays = {(1,1), (5,1), (5,8), (7,14), (8,15), (11,1), (11,11), (12,31)}
    if dt.month == 12 and dt.day not in [24,25] and dt.day >= 17:
        return 1
    if dt.month == 10 and dt.day >= 25:
        return 1
    if dt.month in [9,10] and dt.weekday() == 5:
        return 1
    if (dt.month, dt.day) in fixed_holidays:
        return 1
    return 0

start_covid = pd.Timestamp("2020-03-01 00:00:00")
end_covid   = pd.Timestamp("2021-12-31 23:59:59")

# --- Fonction de préparation du dataframe ---
def modify_df(df):
    weather = pd.read_csv("weather_data.csv")
    df = df.merge(weather, on="DATETIME", how="left")
    #df = pd.get_dummies(df, columns=["ENTITY_DESCRIPTION_SHORT"])
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%Y-%m-%d %H:%M:%S")
    df["is_covid"] = df["DATETIME"].between(start_covid, end_covid).astype(int)
    
    df["day_of_week"] = df["DATETIME"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df = pd.get_dummies(df, columns=["day_of_week"])
    
    df["month"] = df["DATETIME"].dt.month
    df["is_vacation"] = df["month"].isin([6,7,8]).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    df["hour"] = df["DATETIME"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    df["special_event"] = df["DATETIME"].apply(mark_special_event)
    
    df = df.drop(columns=["DATETIME", "hour", "month"])
    df = df.drop(columns=["ENTITY_DESCRIPTION_SHORT"])
    
    # Remplissage des NaN pour éviter problèmes XGBoost
    df = df.fillna(df.median())
    
    return df

attraction = "Water Ride"

train_set = pd.read_csv("waiting_times_train.csv")
#train_set = train_set[train_set["ENTITY_DESCRIPTION_SHORT"] == attraction]
y_target = train_set["WAIT_TIME_IN_2H"]
train_set = train_set.drop(columns=["WAIT_TIME_IN_2H"])
merged_train = modify_df(train_set)

test_set = pd.read_csv("waiting_times_X_test_val.csv")
#test_set = test_set[test_set["ENTITY_DESCRIPTION_SHORT"] == attraction]
merged_test = modify_df(test_set)


X_train, X_val, y_train, y_val = train_test_split(
    merged_train, y_target, test_size=0.2, random_state=42
)


model = XGBRegressor(
    objective='reg:squarederror',
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    n_estimators=2000,
    random_state=42,
    gamma=0.3,
    min_child_weight=6,
    alpha=0.1,
    lambda_=0.5,
    eval_metric='rmse',
    early_stopping_rounds=100,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


y_val_pred = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("Validation RMSE:", rmse_val)


y_test_pred = model.predict(merged_test)
#y_test_pred = np.round(model.predict(merged_test) / 5) * 5  # arrondi au multiple de 5
output = test_set.copy()
output["y_pred"] = y_test_pred
output["KEY"] = "Validation"
cols_arendre_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME"]
output = output.drop(columns=[c for c in cols_arendre_drop if c in output.columns])
output.to_csv("predictions1.csv", index=False)
