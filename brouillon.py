"""
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
SEED=41

train_set = pd.read_csv("waiting_times_train.csv")
train_sans_y = train_set.iloc[:, :-1]
#print(train_sans_y.head())
y_time = train_set.iloc[:, -1]




def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))


def true_train_function(a=0, b=None):

    if b == None:
        b = len(train_sans_y) # b est exclus

    new_X = train_sans_y.iloc[a:b]
    Y = y_time.iloc[a:b]

    return Y

def generate_data(D=4, N=100):
    np.random.seed(SEED)  # Set the seed for reproducibility
    X = np.random.rand(N, D)  # Generate random observation
    Y = np.random.rand(N, 1) # Create target value and add noise
    return X, Y

def poly_fit(X, Y, deg):
    # Generate polynomial features up to the specified degree
    X_poly = PolynomialFeatures(degree=deg).fit_transform(X)


    # Initialize and fit the linear regression model from X_poly to Y
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, Y)

    return lin_reg

# Function to apply the polynomial regression model to new data
def poly_apply(lin_reg, degree, X):
    # Generate polynomial features for the new data
    X_poly = PolynomialFeatures(degree).fit_transform(X)

    # Predict target values using the fitted model
    return lin_reg.predict(X_poly)


# Set the polynomial degree
train_sans_y, y_time = generate_data()
deg = 2
# Fit a polynomial regression model of degree deg to the training data
lin_reg = poly_fit(train_sans_y, y_time, deg)
# Calculate the Root Mean Squared Error (RMSE) for the training and test sets
RMSE_train = RMSE(poly_apply(lin_reg, deg, train_sans_y), y_time)
#RMSE_test = RMSE(poly_apply(lin_reg, deg, X_test), Y_test) #TODO

print(f"Degree = {deg}, RMSE_train = {RMSE_train:.3f}")
"""

"""
# Evaluate RMSE for polynomial degrees from 1 to 8
degrees = range(1, 10)  # Define the range of polynomial degrees to evaluate
RMSE_train_list = []  # List to store RMSE for training data


# Loop through each degree, fit the model, and calculate RMSE
for deg in degrees:
    # Fit the polynomial regression model on the train with the current degree
    lin_reg = poly_fit(train_set_new, y_time, deg)

    # Calculate the Root Mean Squared Error (RMSE) for the training and test sets
    RMSE_train = RMSE(poly_apply(lin_reg, deg,train_set_new ), y_time)


    #print(f"Degree = {deg}, RMSE_train = {RMSE_train:.3f}, RMSE_test = {RMSE_test:.3f}")

    RMSE_train_list.append(RMSE_train)


    #print(f"Degree = {deg}, RMSE_train = {RMSE_train:.3f}, RMSE_test = {RMSE_test:.3f}")

# Plot RMSE for training and test sets across different polynomial degrees
plt.plot(degrees, RMSE_train_list, label='Train RMSE', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.title('RMSE for Training and Test Sets')
plt.legend()
plt.show()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
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

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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
merged_train = merged_train.drop(columns=["DATETIME", "hour", "month"])
merged_train = merged_train.drop(columns=["WAIT_TIME_IN_2H"])
"""
merged_train["temp_diff"] = merged_train["feels_like"] - merged_train["temp"]
merged_train["temp_humidity"] = merged_train["temp"] * merged_train["humidity"]/100
merged_train["wind_effect"] = merged_train["wind_speed"] * (1 + merged_train["clouds_all"]/100)
"""
features_nan = ["TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW", "rain_1h", "snow_1h"]
for col in features_nan:
    #merged_train[col] = merged_train[col].fillna(merged_train[col].mean())
    merged_train[col] = merged_train[col].fillna(merged_train[col].median())
"""
merged_train["precipitation"] = merged_train["rain_1h"].fillna(0) + merged_train["snow_1h"].fillna(0)
merged_train['wait_rolling_mean_3h'] = merged_train['CURRENT_WAIT_TIME'].rolling(3).mean().fillna(merged_train['CURRENT_WAIT_TIME'].median())
merged_train["next_parade_time"] = merged_train[["TIME_TO_PARADE_1","TIME_TO_PARADE_2"]].min(axis=1)
merged_train["parade_in_progress"] = ((merged_train["TIME_TO_PARADE_1"] <= 0) | (merged_train["TIME_TO_PARADE_2"] <= 0)).astype(int)
merged_train["night_show_in_progress"] = (merged_train["TIME_TO_NIGHT_SHOW"] <= 0).astype(int)
merged_train["adjusted_load"] = merged_train["CURRENT_WAIT_TIME"] / (merged_train["ADJUST_CAPACITY"] + 1)
merged_train["total_event_wait"] = merged_train[["TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"]].sum(axis=1)
"""
cols_train_drop = ["dew_point", "pressure", "day_of_week_0", "day_of_week_1", "day_of_week_2", "day_of_week_3", "day_of_week_4", "day_of_week_5", "day_of_week_6"]#, "rain_1h", "snow_1h"]
merged_train = merged_train.drop(columns=[c for c in cols_train_drop if c in merged_train.columns])
scaler = StandardScaler()
print(merged_train.head())
x_train = pd.DataFrame(
    scaler.fit_transform(merged_train),
    columns=merged_train.columns,
    index=merged_train.index
) #x_train_scaled
#x_train = pd.read_csv("waiting_times_train(in).csv", sep=";")

print(x_train.head())
print("Nombre de colonnes :", x_train.shape[1])
print("Noms des colonnes :", list(x_train.columns))

# Convertir DataFrame/Series en tensors
x_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

print("X NaN:", np.isnan(x_tensor.numpy()).sum())
print("X inf:", np.isinf(x_tensor.numpy()).sum())
print("Y NaN:", np.isnan(y_tensor.numpy()).sum())
print("Y inf:", np.isinf(y_tensor.numpy()).sum())

dataset = TensorDataset(x_tensor, y_tensor)

# Séparer train/val (80/20 par exemple)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


input_dim = x_train.shape[1]  # nombre de features
hidden_dim = 128
dropout=0.4
#num_classes = 37

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
    
class MLPImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),                  # ou nn.LeakyReLU(0.1)
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)   # sortie régression
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=25, out_channels=32, kernel_size=3, padding=1),  # entrée = 37 features
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.AdaptiveMaxPool1d(1)  # réduit la séquence à 1 pour chaque canal
        )

        # Fully connected (MLP)
        self.fc_layers = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 1)  # sortie régression
        )

    def forward(self, x):
        # x a la forme (B, features)
        x = x.unsqueeze(2)  # transforme en (B, features, 1) pour Conv1d
        x = self.conv_layers(x)  # (B, 64, 1)
        x = x.view(x.size(0), -1)  # flatten -> (B, 64)
        x = self.fc_layers(x)       # -> (B, 1)
        return x.squeeze(1)
    
model = MLPBaseline()
#model = MLPImproved()
#model = CNNBaseline()

criterion = nn.MSELoss()  # classification multi-classe
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
"""
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # on cherche à minimiser la loss
    factor=0.5,       # lr = lr * factor
    patience=3,       # combien d'epochs sans amélioration avant de réduire
)
"""

num_epochs = 100
best_rmse = float('inf')
iteration = 0
limite_iteration = 10
best_state = None

for epoch in range(num_epochs):
    model.train()
    for x_train_batch, y_train_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_train_batch)
        loss = criterion(outputs, y_train_batch)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    squared_errors = []
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            outputs = model(x_val_batch)
            errors = (outputs - y_val_batch) ** 2
            squared_errors.append(errors.mean().item())

    rmse = np.sqrt(np.mean(squared_errors))
    print(f"Epoch {epoch+1}: Val RMSE = {rmse:.4f}")

    #scheduler.step(rmse)

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
merged_test = merged_test.drop(columns=["DATETIME", "hour", "month"])
"""
merged_test["temp_diff"] = merged_test["feels_like"] - merged_test["temp"]
merged_test["temp_humidity"] = merged_test["temp"] * merged_test["humidity"]/100
merged_test["wind_effect"] = merged_test["wind_speed"] * (1 + merged_test["clouds_all"]/100)
"""
features_nan = ["TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW", "rain_1h", "snow_1h"]
for col in features_nan:
    #merged_test[col] = merged_test[col].fillna(merged_test[col].mean())
    merged_test[col] = merged_test[col].fillna(merged_test[col].median())
"""
merged_test["precipitation"] = merged_test["rain_1h"].fillna(0) + merged_test["snow_1h"].fillna(0)
merged_test['wait_rolling_mean_3h'] = merged_test['CURRENT_WAIT_TIME'].rolling(3).mean().fillna(merged_test['CURRENT_WAIT_TIME'].median())
merged_test["next_parade_time"] = merged_test[["TIME_TO_PARADE_1","TIME_TO_PARADE_2"]].min(axis=1)
merged_test["parade_in_progress"] = ((merged_test["TIME_TO_PARADE_1"] <= 0) | (merged_test["TIME_TO_PARADE_2"] <= 0)).astype(int)
merged_test["night_show_in_progress"] = (merged_test["TIME_TO_NIGHT_SHOW"] <= 0).astype(int)
merged_test["adjusted_load"] = merged_test["CURRENT_WAIT_TIME"] / (merged_test["ADJUST_CAPACITY"] + 1)
merged_test["total_event_wait"] = merged_test[["TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"]].sum(axis=1)
"""
cols_test_drop = ["dew_point", "pressure", "day_of_week_0", "day_of_week_1", "day_of_week_2", "day_of_week_3", "day_of_week_4", "day_of_week_5", "day_of_week_6"]#, "rain_1h", "snow_1h"]
merged_test = merged_test.drop(columns=[c for c in cols_test_drop if c in merged_test.columns])
x_test = pd.DataFrame(
    scaler.transform(merged_test),
    columns=merged_test.columns,
    index=merged_test.index) #x_test_scaled
#x_test = pd.read_csv("waiting_times_X_test_val(2).csv")

x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(x_test_tensor), batch_size=32)

# Prédictions
model.eval()
all_preds = []
with torch.no_grad():
    for (x_test_batch,) in test_loader:
        preds = model(x_test_batch)
        """
        # arrondir aux multiples de 5 pour garder l'échelle
        preds_rounded = torch.clamp(torch.round(preds / 5) * 5, 0, 180)
        all_preds.extend(preds_rounded.cpu().numpy())
        """
        all_preds.extend(preds.cpu().numpy())

y_test_pred = pd.Series(all_preds, name="y_pred")
set_arendre = pd.read_csv("waiting_times_X_test_val.csv")
cols_arendre_drop = ["DOWNTIME", "TIME_TO_PARADE_1", "TIME_TO_PARADE_2", "TIME_TO_NIGHT_SHOW", "ADJUST_CAPACITY", "CURRENT_WAIT_TIME"]
set_arendre = set_arendre.drop(columns=[c for c in cols_arendre_drop if c in set_arendre.columns])
set_arendre["y_pred"] = y_test_pred
set_arendre["KEY"] = "Validation"
set_arendre.to_csv("set_arendre.csv", index=False)
