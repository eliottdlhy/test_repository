import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.datasets import make_classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
# Set random seed for reproducibility
np.random.seed(42)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
import numpy as np


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


weather = pd.read_csv("weather_data.csv")
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
"""
merged_test["temp_diff"] = merged_test["feels_like"] - merged_test["temp"]
merged_test["temp_humidity"] = merged_test["temp"] * merged_test["humidity"]/100
merged_test["wind_effect"] = merged_test["wind_speed"] * (1 + merged_test["clouds_all"]/100)

merged_test["precipitation"] = merged_test["rain_1h"].fillna(0) + merged_test["snow_1h"].fillna(0)
merged_test['wait_rolling_mean_3h'] = merged_test['CURRENT_WAIT_TIME'].rolling(3).mean().fillna(merged_test['CURRENT_WAIT_TIME'].median())
merged_test["next_parade_time"] = merged_test[["TIME_TO_PARADE_1","TIME_TO_PARADE_2"]].min(axis=1)
merged_test["parade_in_progress"] = ((merged_test["TIME_TO_PARADE_1"] <= 0) | (merged_test["TIME_TO_PARADE_2"] <= 0)).astype(int)
merged_test["night_show_in_progress"] = (merged_test["TIME_TO_NIGHT_SHOW"] <= 0).astype(int)
merged_test["adjusted_load"] = merged_test["CURRENT_WAIT_TIME"] / (merged_test["ADJUST_CAPACITY"] + 1)
merged_test["total_event_wait"] = merged_test[["TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"]].sum(axis=1)
"""
"""
cols_test_drop = ["dew_point", "pressure", "day_of_week_0", "day_of_week_1", "day_of_week_2", "day_of_week_3", "day_of_week_4", "day_of_week_5", "day_of_week_6"]#, "rain_1h", "snow_1h"]
merged_test = merged_test.drop(columns=[c for c in cols_test_drop if c in merged_test.columns])
"""


# Features
features = merged_train
target = y_target
x_test = merged_test

# Train/test split
x_train, x_val, y_train, y_val = train_test_split(
    features, target, test_size=0.2, random_state=42
)


dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test)


# 4. Définition des paramètres du modèle
params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.136666666666666666667,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'alpha': 0.1,          # L1 regularization
    'lambda': 0.5,         # L2 regularization
    'random_state': 42,
    'eval_metric': 'rmse',
    "n_estimators": 1000
}
# 5. Entraînement avec validation early stopping
num_round = 100
watchlist = [(dtrain, 'train'), (dval, 'val')]

model = xgb.XGBRegressor(**params)

# Entraînement avec validation set pour early stopping
model.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],
    verbose=10
)

y_train_pred = model.predict(x_train)

y_pred = model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)

all_preds = model.predict(x_test)
"""
# Features
features = merged_train
target = y_target
x_test = merged_test

# Train/test split
x_train, x_val, y_train, y_val = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Définir le modèle de base
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)


param_dist = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": np.linspace(0.01, 0.2, 10),
    "subsample": np.linspace(0.6, 1.0, 5),
    "colsample_bytree": np.linspace(0.6, 1.0, 5),
    "alpha": [0, 0.1, 0.5, 1],
    "lambda": [0.5, 1, 2, 5],
    "n_estimators": [200, 500, 1000, 1500]
}

rand_search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
    param_distributions=param_dist,
    n_iter=30,  # nombre de tirages
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

rand_search.fit(x_train, y_train)
print("Best params:", rand_search.best_params_)
print("Best RMSE:", -rand_search.best_score_)
"""
"""
# Grille d’hyperparamètres à tester
param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "alpha": [0, 0.1, 1],
    "lambda": [0.5, 1, 2],
    "n_estimators": [300, 500, 1000]
}

# GridSearchCV
grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",  # on optimise RMSE
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Entraînement
grid.fit(x_train, y_train)

# Résultats
print("Best params:", grid.best_params_)
print("Best CV RMSE:", -grid.best_score_)


# Évaluation sur validation
best_model = grid.best_estimator_

y_val_pred = best_model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("Validation RMSE:", rmse)

# Prédictions finales sur le test set
all_preds = best_model.predict(x_test)
"""




y_test_pred = pd.Series(all_preds, name="y_pred")
set_arendre = pd.read_csv("waiting_times_X_test_val.csv")
cols_arendre_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME"]
set_arendre = set_arendre.drop(columns=[c for c in cols_arendre_drop if c in set_arendre.columns])
set_arendre["y_pred"] = y_test_pred
set_arendre["KEY"] = "Validation"
set_arendre.to_csv("arendre_xgboost", index=False)
