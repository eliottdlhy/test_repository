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

train_set_new = pd.read_csv("waiting_times_train(in).csv", sep=";")
train_set = pd.read_csv("waiting_times_train.csv")
#train_sans_y = train_set.iloc[:, :-1]
y_time = train_set.iloc[:, -1]



def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))


def true_train_function(a=0, b=None):

    if b == None:
        #b = len(train_sans_y) 
        b = len(train_set_new) 

    #new_X = train_sans_y.iloc[a:b]
    new_X = train_set_new.iloc[a:b]
    Y = y_time.iloc[a:b]

    return Y


def poly_fit(X, Y, deg):
    """
    Fits a polynomial regression model to the data.

    INPUTS:
    X : numpy.ndarray, Input data of shape (N, D)
    Y : numpy.ndarray, Target values of shape (N,).
    deg : int, degree of the polynomial features.

    RETURNS:
    LinearRegression, The fitted linear regression model.
    """
    X_poly = PolynomialFeatures(degree=deg).fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, Y)

    return lin_reg



def poly_apply(lin_reg, degree, X):
    """
    Applies the fitted polynomial regression model to new data.

    INPUTS:
    lin_reg : LinearRegression, The fitted linear regression model.
    degree : int, The degree of the polynomial features used in the model.
    X : numpy.ndarray, Input data to apply the model on, of shape (N, D).

    RETRUNS:
    numpy.ndarray, The predicted target values for the input data.
    """
    X_poly = PolynomialFeatures(degree).fit_transform(X)

    return lin_reg.predict(X_poly)



deg = 7

#lin_reg = poly_fit(train_sans_y, y_time, deg)
lin_reg = poly_fit(train_set_new, y_time, deg)

#RMSE_train = RMSE(poly_apply(lin_reg, deg, train_sans_y), y_time)
RMSE_train = RMSE(poly_apply(lin_reg, deg, train_set_new), y_time)


print(f"Degree = {deg}, RMSE_train = {RMSE_train:.3f}")


x_test = pd.read_csv("waiting_times_X_test_val(2).csv")
y_test = poly_apply(lin_reg, deg, x_test)
x_test["y_test"] = y_test
x_test.to_csv("nouveau_val_set.csv", index=False)