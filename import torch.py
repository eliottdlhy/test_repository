import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---- Fonction pour retirer les outliers uniquement sur la target ----
def remove_outliers_target_only(X, y, factor=2.5):
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (y >= lower) & (y <= upper)
    return X[mask], y[mask]

# ---- Chargement et feature engineering ----
print(xgb.__version__)

train_set = pd.read_csv("waiting_times_train.csv")
y_target = train_set.iloc[:, -1]

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
merged_train = merged_train.drop(columns=["DATETIME", "hour", "month", "WAIT_TIME_IN_2H"])

# Jeu de test
test_set = pd.read_csv("waiting_times_X_test_val.csv")
merged_test = test_set.merge(weather, on="DATETIME", how="left")
merged_test = pd.get_dummies(merged_test, columns=["ENTITY_DESCRIPTION_SHORT"])
merged_test["DATETIME"] = pd.to_datetime(merged_test["DATETIME"], format="%Y-%m-%d %H:%M:%S")
merged_test["is_covid"] = merged_test["DATETIME"].between(start_covid, end_covid).astype(int)
merged_test["day_of_week"] = merged_test["DATETIME"].dt.dayofweek
merged_test["is_weekend"] = merged_test["day_of_week"].isin([5,6]).astype(int)
merged_test = pd.get_dummies(merged_test, columns=["day_of_week"])
merged_test["month"] = merged_test["DATETIME"].dt.month
merged_test["is_vacation"] = merged_test["month"].isin([6,7,8]).astype(int)
merged_test["month_sin"] = np.sin(2 * np.pi * merged_test["month"] / 12)
merged_test["month_cos"] = np.cos(2 * np.pi * merged_test["month"] / 12)
merged_test["hour"] = merged_test["DATETIME"].dt.hour
merged_test["hour_sin"] = np.sin(2 * np.pi * merged_test["hour"] / 24)
merged_test["hour_cos"] = np.cos(2 * np.pi * merged_test["hour"] / 24)
merged_test = merged_test.drop(columns=["DATETIME", "hour", "month"])

# ---- Nettoyage des outliers sur la target uniquement ----
features, target = remove_outliers_target_only(merged_train, y_target, factor=2.5)

x_test = merged_test

# ---- Split ----
x_train, x_val, y_train, y_val = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# ---- XGBoost params ----
params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.06,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0.1,
    'lambda': 0.5,
    'random_state': 42,
    'eval_metric': 'rmse',
    "n_estimators": 295
}

# ---- Entraînement ----
model = xgb.XGBRegressor(**params)
model.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],
    verbose=10
)

# ---- Évaluation ----
y_pred = model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)

# ---- Prédictions ----
all_preds = model.predict(x_test)
y_test_pred = pd.Series(all_preds, name="y_pred")

set_arendre = pd.read_csv("waiting_times_X_test_val.csv")
cols_arendre_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2",
                     "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME"]
set_arendre = set_arendre.drop(columns=[c for c in cols_arendre_drop if c in set_arendre.columns])
set_arendre["y_pred"] = y_test_pred
set_arendre["KEY"] = "Validation"
set_arendre.to_csv("Larendre_xgboost", index=False)
