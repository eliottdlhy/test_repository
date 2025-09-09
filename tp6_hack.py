import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
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

classes = np.linspace(0, 180, 37)

train_set_new = pd.read_csv("waiting_times_train(in).csv", sep=";")

x_test = pd.read_csv("waiting_times_X_test_val(2).csv")

batch_size = 64
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

train_loader = DataLoader(train_set_new, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(x_test, batch_size=batch_size, shuffle=False, num_workers=0)



class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=36):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)

X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.long)  # labels en long !

# Dataset + DataLoader
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)