import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)



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

def modify_df(data_frame):
    weather = pd.read_csv("weather_data.csv")
    df= data_frame.merge(weather, on="DATETIME", how="left")
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
    return df


attraction = "Flying Coaster"
#Train
train_set = pd.read_csv("waiting_times_train.csv")
train_set = train_set[train_set["ENTITY_DESCRIPTION_SHORT"] == attraction]
y_target = train_set["WAIT_TIME_IN_2H"]
train_set = train_set.drop(columns=["WAIT_TIME_IN_2H", "ENTITY_DESCRIPTION_SHORT"])

merged_train = modify_df(train_set)

# Test
test_set = pd.read_csv("waiting_times_X_test_val.csv")
test_set = test_set[test_set["ENTITY_DESCRIPTION_SHORT"] == attraction]
test_set = test_set.drop(columns=["ENTITY_DESCRIPTION_SHORT"])

merged_test = modify_df(test_set)

##########################################################################################################""
# Features
features = merged_train
target = y_target
x_test = merged_test

print(features.shape)
print(x_test.shape)

# Train/test split
x_train, x_val, y_train, y_val = train_test_split(
    features, target, test_size=0.3, random_state=42
)


dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,          # arbres moins profonds
    'learning_rate': 0.05,
    'subsample': 0.7,        # échantillonnage aléatoire
    'colsample_bytree': 0.7, # sélection aléatoire de features
    'min_child_weight': 5,   # plus strict sur les splits
    'gamma': 0.1,            # pénalise les splits faibles
    'alpha': 1,              # régularisation L1
    'lambda': 2,             # régularisation L2
    'n_estimators': 1000,
    'random_state': 42,
    'eval_metric': 'rmse'
}

num_round = 100
watchlist = [(dtrain, 'train'), (dval, 'val')]

model = xgb.XGBRegressor(**params)

model.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],
    verbose=10
)

y_train_pred = model.predict(x_train)
y_pred = model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)


y_test_pred = model.predict(x_test)
set_arendre = test_set.reset_index(drop=True)
set_arendre["ENTITY_DESCRIPTION_SHORT"] = attraction
set_arendre["y_pred"] = y_test_pred
set_arendre["KEY"] = "Validation"
cols_arendre_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME"]
set_arendre = set_arendre.drop(columns=[c for c in cols_arendre_drop if c in set_arendre.columns])
set_arendre.to_csv("arendre_xgboost_fc.csv", index=False)


#############################################################################################################
"""
all_preds = model.predict(x_test)
#all_preds = np.round(all_preds / 5) * 5      #arrondir semble ne pas marcher
y_test_pred = pd.Series(all_preds, name="y_pred")
set_arendre = pd.read_csv("waiting_times_X_test_val.csv")
cols_arendre_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME"]
set_arendre = set_arendre.drop(columns=[c for c in cols_arendre_drop if c in set_arendre.columns])
 
set_arendre = set_arendre[set_arendre["ENTITY_DESCRIPTION_SHORT"] == "Water Ride"] 


set_arendre["y_pred"] = y_test_pred
set_arendre["KEY"] = "Validation"
set_arendre.to_csv("arendre_xgboost_wr", index=False)
"""
"""
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

#################################################################################################
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

#####################################################################################

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

