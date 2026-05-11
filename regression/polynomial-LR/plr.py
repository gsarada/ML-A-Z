#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#load dataset
dataset = pd.read_csv('../Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Training LR
l_regressor = LinearRegression()
l_regressor.fit(X, y)

#Training PLR
pf = PolynomialFeatures(4)
X_new = pf.fit_transform(X)
p_regressor = LinearRegression()
p_regressor.fit(X_new, y)

#Visualize LR results
y_pred = l_regressor.predict(X)
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('Level vs Salary')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

#Visualize PLR results
y_pred1 = p_regressor.predict(X_new)
plt.scatter(X, y, color='red')
plt.plot(X, y_pred1, color='blue')
plt.title('Level vs Salary')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

#Predict new result with LR
y_new = l_regressor.predict(np.array([6.5]).reshape(1, -1))
print(f'Predicted salary for level 6.5 of experience with lr {y_new}')

#Predict new result with PLR
y_new = p_regressor.predict(pf.fit_transform(np.array([6.5]).reshape(1, -1)))
print(f'Predicted salary for level 6.5 of experience with plr {y_new}')
