import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# --- Suppression des outliers uniquement sur la target ---
def remove_outliers_target_only(X, y, factor=2.5):
    # Réinitialiser les index pour aligner
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    mask = (y >= lower) & (y <= upper)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

# --- Marquage événements spéciaux ---
def mark_special_event(dt):
    fixed_holidays = {(1,1), (5,1), (5,8), (7,14), (8,15), (11,1), (11,11), (12,31)}
    if dt.month == 12 and dt.day not in [24,25] and dt.day >= 17: return 1
    if dt.month == 10 and dt.day >= 25: return 1
    if dt.month in [9,10] and dt.weekday() == 5: return 1
    if (dt.month, dt.day) in fixed_holidays: return 1
    return 0

start_covid = pd.Timestamp("2020-03-01 00:00:00")
end_covid   = pd.Timestamp("2021-12-31 23:59:59")

# --- Préparation des features ---
def modify_df(df):
    try:
        weather = pd.read_csv("weather_data.csv")
        df = df.merge(weather, on="DATETIME", how="left")
    except FileNotFoundError:
        pass

    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["is_covid"] = df["DATETIME"].between(start_covid, end_covid).astype(int)
    df["day_of_week"] = df["DATETIME"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["month"] = df["DATETIME"].dt.month
    df["day"] = df["DATETIME"].dt.day
    df["hour"] = df["DATETIME"].dt.hour
    df["day_of_year"] = df["DATETIME"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["is_summer"] = df["month"].isin([6,7,8]).astype(int)
    df["is_winter"] = df["month"].isin([12,1,2]).astype(int)
    df["is_spring"] = df["month"].isin([3,4,5]).astype(int)
    df["is_fall"] = df["month"].isin([9,10,11]).astype(int)

    df["is_vacation"] = ((df["month"].isin([7,8])) |
                         ((df["month"] == 12) & (df["day"] >= 20)) |
                         ((df["month"] == 2) & (df["day"] <= 7))).astype(int)

    df["special_event"] = df["DATETIME"].apply(mark_special_event)
    df["weekend_hour"] = df["is_weekend"] * df["hour"]
    df["vacation_hour"] = df["is_vacation"] * df["hour"]

    cols_to_drop = ["DATETIME", "hour", "month", "day", "day_of_year"]
    if "ENTITY_DESCRIPTION_SHORT" in df.columns:
        cols_to_drop.append("ENTITY_DESCRIPTION_SHORT")

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.dropna(axis=1, thresh=len(df)*0.8)

    return df

# --- Chargement ---
train_set = pd.read_csv("waiting_times_train.csv")
test_set = pd.read_csv("waiting_times_X_test_val.csv")
y_target = train_set["WAIT_TIME_IN_2H"]

attractions = train_set["ENTITY_DESCRIPTION_SHORT"].unique()
print(f"{len(attractions)} attractions trouvées.")

all_predictions = []

for attraction in attractions:
    print(f"\n=== Traitement: {attraction} ===")
    train_sub = train_set[train_set["ENTITY_DESCRIPTION_SHORT"] == attraction].copy()
    test_sub = test_set[test_set["ENTITY_DESCRIPTION_SHORT"] == attraction].copy()
    y_sub = y_target[train_set["ENTITY_DESCRIPTION_SHORT"] == attraction].copy().reset_index(drop=True)

    if len(train_sub) < 20:
        print(f"⚠️ Trop peu de données pour {attraction}, ignoré.")
        continue

    merged_train = modify_df(train_sub)
    merged_test = modify_df(test_sub)
    common_cols = merged_train.columns.intersection(merged_test.columns)
    merged_train = merged_train[common_cols]
    merged_test = merged_test[common_cols]

    # Suppression des outliers
    X_clean, y_clean = remove_outliers_target_only(merged_train, y_sub, factor=2.5)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, shuffle=True
    )

    # Modèle
    model = XGBRegressor(
        objective='reg:squarederror',
        max_depth=5,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=700,
        random_state=42,
        gamma=0.2,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='rmse',
        early_stopping_rounds=50000,
        n_jobs=-1,
        tree_method='hist'
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    y_val_pred = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation RMSE ({attraction}): {rmse_val:.3f}")

    preds = model.predict(merged_test)
    out = test_sub.copy()
    out["y_pred"] = preds
    out["ATTRACTION"] = attraction
    all_predictions.append(out)

final_output = pd.concat(all_predictions, ignore_index=True)
cols_arendre_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2",
                     "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME", "ATTRACTION"]
final_output = final_output.drop(columns=[c for c in cols_arendre_drop if c in final_output.columns])
final_output["KEY"] = "Validation"
final_output.to_csv("predictions_per_entity_no_outliers.csv", index=False)
print("\n Prédictions sauvegardées dans 'predictions_per_entity_no_outliers.csv'")



