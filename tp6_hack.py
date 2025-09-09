import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

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

x_train = pd.read_csv("waiting_times_train(in).csv", sep=";")

train_set = pd.read_csv("waiting_times_train.csv")
y_train = train_set.iloc[:, -1]


# Convertir DataFrame/Series en tensors
x_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

dataset = TensorDataset(x_tensor, y_tensor)

# Séparer train/val (80/20 par exemple)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


input_dim = x_train.shape[1]  # nombre de features
hidden_dim = 64
dropout=0.4

class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)
    
model = MLPBaseline()

criterion = nn.MSELoss()  # classification multi-classe
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


num_epochs = 100
best_rmse = float('inf')
iteration = 0
limite_iteration = 5
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


x_test = pd.read_csv("waiting_times_X_test_val(2).csv")
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
test_loader = DataLoader(x_test_tensor, batch_size=32)

# Prédictions
model.eval()
all_preds = []
with torch.no_grad():
    for x_test_batch in test_loader:
        preds = model(x_test_batch)
        # arrondir aux multiples de 5 pour garder l'échelle
        preds_rounded = torch.clamp(torch.round(preds / 5) * 5, 0, 180)
        all_preds.extend(preds_rounded.cpu().numpy())

y_test_pred = pd.Series(all_preds, name="y_pred")
set_arendre = pd.read_csv("waiting_times_X_test_val.csv")
set_arendre = set_arendre.drop("ADJUST_CAPACITY", axis=1)
set_arendre = set_arendre.drop("DOWNTIME", axis=1)
set_arendre = set_arendre.drop("CURRENT_WAIT_TIME", axis=1)
set_arendre = set_arendre.drop("TIME_TO_PARADE_1", axis=1)
set_arendre = set_arendre.drop("TIME_TO_PARADE_2", axis=1)
set_arendre = set_arendre.drop("TIME_TO_NIGHT_SHOW", axis=1)
set_arendre["KEY"] = "Validation"
set_arendre["y_pred"] = y_test_pred
set_arendre.to_csv("set_arendre", index=False)