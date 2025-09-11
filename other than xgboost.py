from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import random
import optuna
warnings.filterwarnings('ignore')

np.random.seed(42)


def remove_outliers_target_only(X, y, factor=2.5):
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (y >= lower) & (y <= upper)
    return X[mask], y[mask]


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

# --- Fonction de préparation du dataframe améliorée ---
def modify_df(df):
    # Chargement des données météo avec gestion d'erreur
    try:
        weather = pd.read_csv("weather_data.csv")
        df = df.merge(weather, on="DATETIME", how="left")
        print(f"Météo fusionnée: {weather.shape[1]} colonnes ajoutées")
    except FileNotFoundError:
        print("Fichier weather_data.csv non trouvé, continuation sans données météo")
    
    #df = pd.get_dummies(df, columns=["ENTITY_DESCRIPTION_SHORT"])
    # Conversion datetime
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%Y-%m-%d %H:%M:%S")
    
    # Features temporelles avancées
    df["is_covid"] = df["DATETIME"].between(start_covid, end_covid).astype(int)
    df["day_of_week"] = df["DATETIME"].dt.dayofweek
    
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    #df = pd.get_dummies(df, columns=["day_of_week"])
    df["month"] = df["DATETIME"].dt.month
    df["day"] = df["DATETIME"].dt.day
    df["hour"] = df["DATETIME"].dt.hour
    df["day_of_year"] = df["DATETIME"].dt.dayofyear
    
    # Features cycliques
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    
    # Saisons
    df["is_summer"] = df["month"].isin([6,7,8]).astype(int)
    df["is_winter"] = df["month"].isin([12,1,2]).astype(int)
    df["is_spring"] = df["month"].isin([3,4,5]).astype(int)
    df["is_fall"] = df["month"].isin([9,10,11]).astype(int)
    
    # Périodes de vacances (approximatif)
    df["is_vacation"] = ((df["month"].isin([7,8])) | 
                         ((df["month"] == 12) & (df["day"] >= 20)) |
                         ((df["month"] == 2) & (df["day"] <= 7))).astype(int)
    
    # Événements spéciaux
    df["special_event"] = df["DATETIME"].apply(mark_special_event)

    # Encodage de l'attraction si présente
    if "ENTITY_DESCRIPTION_SHORT" in df.columns:
        le = LabelEncoder()
        df["attraction_encoded"] = le.fit_transform(df["ENTITY_DESCRIPTION_SHORT"])
        print(f"Attractions encodées: {len(le.classes_)} attractions différentes")

    
    """
    # Features d'interaction
    df["weekend_hour"] = df["is_weekend"] * df["hour"]
    df["vacation_hour"] = df["is_vacation"] * df["hour"]
    
    # --- Création de nouvelles features météorologiques et opérationnelles ---
    
    df["temp_humidity"] = df["temp"] * df["humidity"] / 100
    df["wind_effect"] = df["wind_speed"] * (1 + df["clouds_all"] / 100)

    # Précipitations totales
    df["precipitation"] = df["rain_1h"].fillna(0) + df["snow_1h"].fillna(0)

    # Rolling mean de l'attente
    df['wait_rolling_mean_3h'] = df['CURRENT_WAIT_TIME'].rolling(3).mean().fillna(df['CURRENT_WAIT_TIME'].median())

    # Temps avant le prochain défilé et shows
    df["next_parade_time"] = df[["TIME_TO_PARADE_1", "TIME_TO_PARADE_2"]].min(axis=1)
    df["parade_in_progress"] = ((df["TIME_TO_PARADE_1"] <= 0) | (df["TIME_TO_PARADE_2"] <= 0)).astype(int)
    df["night_show_in_progress"] = (df["TIME_TO_NIGHT_SHOW"] <= 0).astype(int)

    # Charge ajustée et total événementiel
    df["adjusted_load"] = df["CURRENT_WAIT_TIME"] / (df["ADJUST_CAPACITY"] + 1)
    """
    #df["total_event_wait"] = df[["TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW"]].sum(axis=1)

    # Drop des colonnes inutiles
    cols_to_drop = ["DATETIME"]
    if "ENTITY_DESCRIPTION_SHORT" in df.columns:
        cols_to_drop.append("ENTITY_DESCRIPTION_SHORT")
    
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df["temp_diff"] = df["feels_like"] - df["temp"]
    # Remplissage des NaN

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Suppression des colonnes avec trop de NaN
    #df = df.dropna(axis=1, thresh=len(df)*0.8)
    
    print(f"DataFrame final shape: {df.shape}")
    
    return df

# --- Chargement et préparation des données ---
print("Chargement des données...")
train_set = pd.read_csv("waiting_times_train.csv")
test_set = pd.read_csv("waiting_times_X_test_val.csv")

# Filtrer pour une attraction spécifique ou garder toutes
attraction = None  # Mettez None pour toutes les attractions
if attraction:
    train_set = train_set[train_set["ENTITY_DESCRIPTION_SHORT"] == attraction]
    test_set = test_set[test_set["ENTITY_DESCRIPTION_SHORT"] == attraction]
    print(f"Filtré pour l'attraction: {attraction}")

y_target = train_set["WAIT_TIME_IN_2H"]
train_set = train_set.drop(columns=["WAIT_TIME_IN_2H"])

print("Préprocessing des données...")
merged_train = modify_df(train_set)
merged_test = modify_df(test_set)



# Alignement des colonnes entre train et test
common_cols = merged_train.columns.intersection(merged_test.columns)
merged_train = merged_train[common_cols]
merged_test = merged_test[common_cols]

print(f"Colonnes finales: {len(common_cols)}")

# ---- Nettoyage des outliers sur la target uniquement ----
features, target = remove_outliers_target_only(merged_train, y_target, factor=2.5)

# --- Split des données avec validation temporelle ---
# Pour les séries temporelles, mieux vaut utiliser un split temporel
X_train, X_val, y_train, y_val = train_test_split(
    merged_train, y_target, test_size=0.5, random_state=42
)


"""
def objective(trial):
    # XGB
    xgb_params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.2),
        max_depth=trial.suggest_int("xgb_max_depth", 3, 8),
        subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),
        n_estimators=trial.suggest_int("xgb_n_estimators", 200, 1000),
        min_child_weight=trial.suggest_int("xgb_min_child_weight", 1, 10),
        gamma=trial.suggest_float("xgb_gamma", 0.0, 0.4),
        tree_method="hist",
        random_state=42
    )
    xgb_model = XGBRegressor(**xgb_params)

    # LightGBM
    lgb_params = dict(
        learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.2),
        num_leaves=trial.suggest_int("lgb_num_leaves", 20, 90),
        feature_fraction=trial.suggest_float("lgb_feature_fraction", 0.6, 1.0),
        bagging_fraction=trial.suggest_float("lgb_bagging_fraction", 0.6, 1.0),
        min_child_samples=trial.suggest_int("lgb_min_child_samples", 10, 60),
        n_estimators=trial.suggest_int("lgb_n_estimators", 200, 1000),
        objective="regression",
        metric="rmse",
        random_state=42
    )
    lgb_model = LGBMRegressor(**lgb_params)

    # Random Forest
    rf_params = dict(
        n_estimators=trial.suggest_int("rf_n_estimators", 100, 500),
        max_depth=trial.suggest_int("rf_max_depth", 5, 20),
        min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 5),
        random_state=42,
        n_jobs=-1
    )
    rf_model = RandomForestRegressor(**rf_params)

    # CatBoost
    cat_params = dict(
        iterations=trial.suggest_int("cat_iterations", 300, 1000),
        depth=trial.suggest_int("cat_depth", 4, 10),
        learning_rate=trial.suggest_float("cat_lr", 0.01, 0.2),
        l2_leaf_reg=trial.suggest_float("cat_l2", 1, 10),
        random_seed=42,
        verbose=0
    )
    cat_model = CatBoostRegressor(**cat_params)

    # Stacking avec Ridge comme méta-modèle
    stack_model = StackingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model),
            ('cat', cat_model)
        ],
        final_estimator=Ridge(alpha=trial.suggest_float("ridge_alpha", 0.1, 10.0)),
        n_jobs=-1
    )

    stack_model.fit(X_train, y_train)
    preds = stack_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


# Optuna

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # Ajuste le nombre d'essais selon ton temps

print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)


# Entraînement final sur toutes les données
best = study.best_params

final_stack = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            learning_rate=best["xgb_lr"],
            max_depth=best["xgb_max_depth"],
            subsample=best["xgb_subsample"],
            colsample_bytree=best["xgb_colsample"],
            n_estimators=best["xgb_n_estimators"],
            min_child_weight=best["xgb_min_child_weight"],
            gamma=best["xgb_gamma"]
        )),
        ('lgb', LGBMRegressor(
            learning_rate=best["lgb_lr"],
            num_leaves=best["lgb_num_leaves"],
            feature_fraction=best["lgb_feature_fraction"],
            bagging_fraction=best["lgb_bagging_fraction"],
            min_child_samples=best["lgb_min_child_samples"],
            n_estimators=best["lgb_n_estimators"],
            objective="regression",
            metric="rmse",
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=best["rf_n_estimators"],
            max_depth=best["rf_max_depth"],
            min_samples_split=best["rf_min_samples_split"],
            min_samples_leaf=best["rf_min_samples_leaf"],
            random_state=42,
            n_jobs=-1
        )),
        ('cat', CatBoostRegressor(
            iterations=best["cat_iterations"],
            depth=best["cat_depth"],
            learning_rate=best["cat_lr"],
            l2_leaf_reg=best["cat_l2"],
            random_seed=42,
            verbose=0
        ))
    ],
    final_estimator=Ridge(alpha=best["ridge_alpha"]),
    n_jobs=-1
)
final_stack = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            learning_rate=0.16046889325937652,
            max_depth=6,
            subsample=0.7609822830479911,
            colsample_bytree=0.7623569781460905,
            n_estimators=400,
            min_child_weight=8,
            gamma=0.382624621563869
        )),
        ('lgb', LGBMRegressor(
            learning_rate=0.06545841885199752,
            num_leaves=89,
            feature_fraction=0.8098527647231846,
            bagging_fraction=0.7161587586765581,
            min_child_samples=47,
            n_estimators=400,
            objective="regression",
            metric="rmse",
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=400,
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )),
        ('cat', CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.12321205339334988,
            l2_leaf_reg=8.7105959731953,
            random_seed=42,
            verbose=0
        ))
    ],
    final_estimator=Ridge(alpha=7.366282594616848),
    n_jobs=-1
)
"""

final_stack = StackingRegressor(
    estimators=[
        ('lgb', LGBMRegressor(
            learning_rate=0.05,
            num_leaves=15,
            feature_fraction=0.4,
            bagging_fraction=0.4,
            min_child_samples=15,
            n_estimators=1000,
            objective="regression",
            metric="rmse",
            random_state=42
        ))
    ],
    final_estimator=Ridge(alpha=4),
    n_jobs=-1
)

final_stack.fit(merged_train, y_target)


# --- Évaluation détaillée ---
print("\n" + "="*50)
print("ÉVALUATION DU MODÈLE")
print("="*50)

y_val_pred = final_stack.predict(X_val)
y_train_pred = final_stack.predict(X_train)

# Métriques de validation
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

# Métriques d'entraînement
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

print(f"Validation RMSE: {rmse_val:.4f}")
print(f"Validation MAE: {mae_val:.4f}")
print(f"Validation R²: {r2_val:.4f}")
print(f"Train RMSE: {rmse_train:.4f}")
print(f"Train R²: {r2_train:.4f}")

# Check overfitting
if rmse_train < rmse_val * 0.7:
    print("⚠️  Attention: possible overfitting (écart important entre train et validation)")

"""
# --- Feature Importance ---
print("\nTop 10 Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_stack.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
"""
# --- Prédiction sur le test set ---
print("\nPrédiction sur le test set...")
y_test_pred = final_stack.predict(merged_test)

# Post-processing: arrondi aux multiples de 5 avec clipping pour éviter les valeurs négatives
#y_test_pred = np.round(np.clip(y_test_pred, 0, None) / 5) * 5

# --- Préparation du fichier de sortie ---
output = test_set.copy()
output["y_pred"] = y_test_pred
output["KEY"] = "Validation"

# Colonnes à drop
cols_to_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", 
                "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME"]
cols_to_drop = [c for c in cols_to_drop if c in output.columns]

output = output.drop(columns=cols_to_drop)

# Vérification des prédictions
print(f"\nStatistiques des prédictions:")
print(f"Min: {y_test_pred.min():.1f}")
print(f"Max: {y_test_pred.max():.1f}")
print(f"Moyenne: {y_test_pred.mean():.1f}")
print(f"Nombre de prédictions: {len(y_test_pred)}")

# Sauvegarde
output.to_csv("predictions_lionel_outlrs.csv", index=False)
print("\n✅ Prédictions sauvegardées dans 'predictions_optimized.csv'")

