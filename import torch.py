import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import random
import pandas as pd

# --- Reproductibilité ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# --- Chargement et fusion des données d'entraînement ---
train_set = pd.read_csv("waiting_times_train.csv")
y_train = train_set.iloc[:, -1]
weather = pd.read_csv("weather_data.csv")
merged_train = train_set.merge(weather, on="DATETIME", how="left")
merged_train = pd.get_dummies(merged_train, columns=["ENTITY_DESCRIPTION_SHORT"])
merged_train["DATETIME"] = pd.to_datetime(merged_train["DATETIME"], format="%Y-%m-%d %H:%M:%S")
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
merged_train["special_event"] = merged_train["DATETIME"].apply(mark_special_event)

merged_train = merged_train.drop(columns=["DATETIME", "hour", "month", "WAIT_TIME_IN_2H"])
features_nan = ["TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW", "rain_1h", "snow_1h"]
for col in features_nan:
    merged_train[col] = merged_train[col].fillna(merged_train[col].median())

cols_train_drop = ["dew_point", "pressure", "day_of_week_0", "day_of_week_1", "day_of_week_2",
                   "day_of_week_3", "day_of_week_4", "day_of_week_5", "day_of_week_6"]
merged_train = merged_train.drop(columns=[c for c in cols_train_drop if c in merged_train.columns])

# --- Mise à l'échelle ---
scaler = StandardScaler()
x_train = pd.DataFrame(
    scaler.fit_transform(merged_train),
    columns=merged_train.columns,
    index=merged_train.index
)

# --- Conversion en tenseurs ---
x_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

dataset = TensorDataset(x_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# --- Modèles ---
input_dim = x_train.shape[1]
hidden_dim = 128
dropout = 0.4

class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)

model = MLPBaseline()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# --- Entraînement avec early stopping ---
num_epochs = 150
best_rmse = float('inf')
iteration = 0
limite_iteration = 10
best_state = None

for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    model.eval()
    squared_errors = []
    with torch.no_grad():
        for xb, yb in val_loader:
            errors = (model(xb) - yb) ** 2
            squared_errors.append(errors.mean().item())
    rmse = np.sqrt(np.mean(squared_errors))
    print(f"Epoch {epoch+1}: Val RMSE = {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_state = model.state_dict()
        iteration = 0
    else:
        iteration += 1
        if iteration >= limite_iteration:
            print("Early stopping !")
            break

model.load_state_dict(best_state)

# --- Prétraitement du test ---
weather = pd.read_csv("weather_data.csv")
test_set = pd.read_csv("waiting_times_X_test_val.csv")
merged_test = test_set.merge(weather, on="DATETIME", how="left")
merged_test = pd.get_dummies(merged_test, columns=["ENTITY_DESCRIPTION_SHORT"])
merged_test["DATETIME"] = pd.to_datetime(merged_test["DATETIME"], format="%Y-%m-%d %H:%M:%S")
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
merged_test["special_event"] = merged_test["DATETIME"].apply(mark_special_event)

merged_test = merged_test.drop(columns=["DATETIME", "hour", "month"])
for col in features_nan:
    merged_test[col] = merged_test[col].fillna(merged_test[col].median())
cols_test_drop = ["dew_point", "pressure", "day_of_week_0", "day_of_week_1", "day_of_week_2",
                  "day_of_week_3", "day_of_week_4", "day_of_week_5", "day_of_week_6"]
merged_test = merged_test.drop(columns=[c for c in cols_test_drop if c in merged_test.columns])

x_test = pd.DataFrame(scaler.transform(merged_test), columns=merged_test.columns, index=merged_test.index)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(x_test_tensor), batch_size=32)

# --- Prédictions ---
model.eval()
all_preds = []
with torch.no_grad():
    for (xb,) in test_loader:
        preds = model(xb)
        all_preds.extend(preds.cpu().numpy())

y_test_pred = pd.Series(all_preds, name="y_pred")
set_arendre = pd.read_csv("waiting_times_X_test_val.csv")
cols_arendre_drop = ["DOWNTIME","TIME_TO_PARADE_1","TIME_TO_PARADE_2",
                     "TIME_TO_NIGHT_SHOW","ADJUST_CAPACITY","CURRENT_WAIT_TIME"]
set_arendre = set_arendre.drop(columns=[c for c in cols_arendre_drop if c in set_arendre.columns])
set_arendre["y_pred"] = y_test_pred
set_arendre["KEY"] = "Validation"
set_arendre.to_csv("set_arendre.csv", index=False)
