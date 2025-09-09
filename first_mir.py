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
train_sans_y = train_set.iloc[:, :-1]
#print(train_sans_y.head())
y_time = train_set.iloc[:, -1]
#print(y_time)
print("début")


def RMSE(x,y):
  return np.sqrt(mean_squared_error(x,y))


def true_train_function(a=0, b=None):

    if b == None:
        #b = len(train_sans_y) # b est exclus
        b = len(train_set_new) # b est exclus

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
    # Generate polynomial features up to the specified degree
    X_poly = PolynomialFeatures(degree=deg).fit_transform(X)


    # Initialize and fit the linear regression model from X_poly to Y
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, Y)

    return lin_reg

# Function to apply the polynomial regression model to new data
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
    # Generate polynomial features for the new data
    X_poly = PolynomialFeatures(degree).fit_transform(X)

    # Predict target values using the fitted model
    return lin_reg.predict(X_poly)


# Set the polynomial degree
deg = 1
# Fit a polynomial regression model of degree deg to the training data
#lin_reg = poly_fit(train_sans_y, y_time, deg)
lin_reg = poly_fit(train_set_new, y_time, deg)
# Calculate the Root Mean Squared Error (RMSE) for the training and test sets
#RMSE_train = RMSE(poly_apply(lin_reg, deg, train_sans_y), y_time)
RMSE_train = RMSE(poly_apply(lin_reg, deg, train_set_new), y_time)
#RMSE_test = RMSE(poly_apply(lin_reg, deg, X_test), Y_test) #TODO

print(f"Degree = {deg}, RMSE_train = {RMSE_train:.3f}")



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

