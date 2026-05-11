#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

#load dataset
dataset = pd.read_csv('../Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature scaling is not required here because the decision tree model simply groups the features so the ranges does not matter
#Train decision tree model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

#Predict a new value
y_new = regressor.predict([[6.5]])
print(y_new)

#Visualize the results
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Level vs Salary')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

#Evaluate the performance
score = r2_score(y, regressor.predict(X))
print(score)
