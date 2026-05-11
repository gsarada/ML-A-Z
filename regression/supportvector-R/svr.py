#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#load dataset
dataset = pd.read_csv('../Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Feature scaling
scaler = StandardScaler()
scaler_y = StandardScaler()
X = scaler.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(len(y), 1)).reshape(len(y))

#Training the SVR model
regressor = SVR(kernel='rbf', epsilon=0.05)
regressor.fit(X, y)

#Predicting a new result
y_new = scaler_y.inverse_transform(regressor.predict(scaler.transform([[6.5]])).reshape(-1, 1))
print(y_new)

#Visualizing the results
plt.scatter(scaler.inverse_transform(X), scaler_y.inverse_transform(y.reshape(1, -1)), color='red')
plt.plot(scaler.inverse_transform(X), scaler_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title('Level vs Salary')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

#Visualization with high resolution
x_grid = np.arange(min(scaler.inverse_transform(X)), max(scaler.inverse_transform(X)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(scaler.inverse_transform(X), scaler_y.inverse_transform(y.reshape(1, len(y))), color='red')
plt.plot(x_grid, scaler_y.inverse_transform(regressor.predict(scaler.transform(x_grid)).reshape(-1, 1)), color='blue')
plt.title('Level vs Salary')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()
