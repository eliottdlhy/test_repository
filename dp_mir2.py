import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

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

# --- Fonction de préparation du dataframe améliorée ---
def modify_df(df):
    # Chargement des données météo avec gestion d'erreur
    try:
        weather = pd.read_csv("weather_data.csv")
        df = df.merge(weather, on="DATETIME", how="left")
        print(f"Météo fusionnée: {weather.shape[1]} colonnes ajoutées")
    except FileNotFoundError:
        print("Fichier weather_data.csv non trouvé, continuation sans données météo")
    
    # Conversion datetime
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%Y-%m-%d %H:%M:%S")
    
    # Features temporelles avancées
    df["is_covid"] = df["DATETIME"].between(start_covid, end_covid).astype(int)
    df["day_of_week"] = df["DATETIME"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
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
    
    # Features d'interaction
    df["weekend_hour"] = df["is_weekend"] * df["hour"]
    df["vacation_hour"] = df["is_vacation"] * df["hour"]
    
    # Drop des colonnes inutiles
    cols_to_drop = ["DATETIME", "hour", "month", "day", "day_of_year"]
    if "ENTITY_DESCRIPTION_SHORT" in df.columns:
        cols_to_drop.append("ENTITY_DESCRIPTION_SHORT")
    
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Remplissage des NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Suppression des colonnes avec trop de NaN
    df = df.dropna(axis=1, thresh=len(df)*0.8)
    
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

# --- Split des données avec validation temporelle ---
# Pour les séries temporelles, mieux vaut utiliser un split temporel
X_train, X_val, y_train, y_val = train_test_split(
    merged_train, y_target, test_size=0.2, random_state=42, shuffle=True
)

# Alternative: Time Series Split (décommenter si données temporelles)
# tscv = TimeSeriesSplit(n_splits=5)
# for train_index, val_index in tscv.split(merged_train):
#     X_train, X_val = merged_train.iloc[train_index], merged_train.iloc[val_index]
#     y_train, y_val = y_target.iloc[train_index], y_target.iloc[val_index]

# --- Modèle XGBoost optimisé ---
print("Entraînement du modèle...")

rf_model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_val_pred_rf = rf_model.predict(X_val)
rmse_val_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
print(f"Random Forest RMSE: {rmse_val_rf:.4f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_val_pred_ridge = ridge_model.predict(X_val_scaled)
rmse_val_ridge = np.sqrt(mean_squared_error(y_val, y_val_pred_ridge))
print(f"Ridge RMSE: {rmse_val_ridge:.4f}")



# --- Préparation du fichier de sortie ---
y_test_pred = rf_model.predict(merged_test)
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
output.to_csv("predictions_optimized.csv", index=False)
print("\n✅ Prédictions sauvegardées dans 'predictions_optimized.csv'")
