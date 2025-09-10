import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---- Fonction pour retirer les outliers uniquement sur la target ----
def remove_outliers_target_only(X, y, factor=1.5):
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (y >= lower) & (y <= upper)
    return X[mask], y[mask]

# ---- Chargement des données ----
train_set = pd.read_csv("waiting_times_train.csv")
y_target = train_set.iloc[:, -1]

weather = pd.read_csv("weather_data.csv")

# ---- Merge et feature engineering ----
merged_train = train_set.merge(weather, on="DATETIME", how="left")
merged_train["DATETIME"] = pd.to_datetime(merged_train["DATETIME"], format="%Y-%m-%d %H:%M:%S")

# Dates covid
start_covid = pd.Timestamp("2020-03-01 00:00:00")
end_covid   = pd.Timestamp("2021-12-31 23:59:59")
merged_train["is_covid"] = merged_train["DATETIME"].between(start_covid, end_covid).astype(int)

# Jour de la semaine, weekend
merged_train["day_of_week"] = merged_train["DATETIME"].dt.dayofweek
merged_train["is_weekend"] = merged_train["day_of_week"].isin([5,6]).astype(int)

# Mois, vacances, trigonométrie
merged_train["month"] = merged_train["DATETIME"].dt.month
merged_train["is_vacation"] = merged_train["month"].isin([6,7,8]).astype(int)
merged_train["month_sin"] = np.sin(2 * np.pi * merged_train["month"] / 12)
merged_train["month_cos"] = np.cos(2 * np.pi * merged_train["month"] / 12)
merged_train["hour"] = merged_train["DATETIME"].dt.hour
merged_train["hour_sin"] = np.sin(2 * np.pi * merged_train["hour"] / 24)
merged_train["hour_cos"] = np.cos(2 * np.pi * merged_train["hour"] / 24)

# One-hot encoding
merged_train = pd.get_dummies(merged_train, columns=["ENTITY_DESCRIPTION_SHORT", "day_of_week"])

# Drop colonnes inutiles
merged_train = merged_train.drop(columns=["DATETIME", "hour", "month", "WAIT_TIME_IN_2H"])

# ---- Nettoyage des outliers ----
features, target = remove_outliers_target_only(merged_train, y_target, factor=2.5)

# ---- Split par entity description ----
train_list = []
val_list = []

# Récupération de la colonne entity avant drop
features["entity_group"] = train_set.loc[features.index, "ENTITY_DESCRIPTION_SHORT"]

for entity, group in features.groupby("entity_group"):
    y_group = target.loc[group.index]
    if len(group) > 1:
        x_tr, x_v = train_test_split(group, test_size=0.2, random_state=42)
        y_tr = y_group.loc[x_tr.index]
        y_v = y_group.loc[x_v.index]
        train_list.append((x_tr, y_tr))
        val_list.append((x_v, y_v))
    else:
        # une seule ligne → train
        train_list.append((group, y_group))

x_train = pd.concat([x for x, y in train_list]).reset_index(drop=True)
y_train = pd.concat([y for x, y in train_list]).reset_index(drop=True)
x_val = pd.concat([x for x, y in val_list]).reset_index(drop=True)
y_val = pd.concat([y for x, y in val_list]).reset_index(drop=True)

# Drop colonne temporaire
x_train = x_train.drop(columns=["entity_group"])
x_val = x_val.drop(columns=["entity_group"])

# ---- Préparation du jeu de test ----
merged_test = pd.read_csv("waiting_times_X_test_val.csv").merge(weather, on="DATETIME", how="left")
merged_test["DATETIME"] = pd.to_datetime(merged_test["DATETIME"], format="%Y-%m-%d %H:%M:%S")
merged_test["is_covid"] = merged_test["DATETIME"].between(start_covid, end_covid).astype(int)
merged_test["day_of_week"] = merged_test["DATETIME"].dt.dayofweek
merged_test["is_weekend"] = merged_test["day_of_week"].isin([5,6]).astype(int)
merged_test["month"] = merged_test["DATETIME"].dt.month
merged_test["is_vacation"] = merged_test["month"].isin([6,7,8]).astype(int)
merged_test["month_sin"] = np.sin(2 * np.pi * merged_test["month"] / 12)
merged_test["month_cos"] = np.cos(2 * np.pi * merged_test["month"] / 12)
merged_test["hour"] = merged_test["DATETIME"].dt.hour
merged_test["hour_sin"] = np.sin(2 * np.pi * merged_test["hour"] / 24)
merged_test["hour_cos"] = np.cos(2 * np.pi * merged_test["hour"] / 24)

# One-hot encoding
merged_test = pd.get_dummies(merged_test, columns=["ENTITY_DESCRIPTION_SHORT", "day_of_week"])
merged_test = merged_test.drop(columns=["DATETIME", "hour", "month"])

# Alignement colonnes train/test
merged_test = merged_test.reindex(columns=x_train.columns, fill_value=0)
x_test = merged_test

# ---- XGBoost params ----
params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.13666666666666667,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'alpha': 0.1,
    'lambda': 0.5,
    'random_state': 42,
    'eval_metric': 'rmse',
    "n_estimators": 1000
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
y_test_pred = pd.Series(model.predict(x_test), name="y_pred")

set_arendre = pd.read_csv("waiting_times_X_test_val.csv")
cols_arendre_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2",
                     "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME"]
set_arendre = set_arendre.drop(columns=[c for c in cols_arendre_drop if c in set_arendre.columns])
set_arendre["y_pred"] = y_test_pred
set_arendre["KEY"] = "Validation"
set_arendre.to_csv("Larendre_xgboost.csv", index=False)
