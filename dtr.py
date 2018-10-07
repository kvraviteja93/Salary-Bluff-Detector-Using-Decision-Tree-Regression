#Importing The Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing The Dataset
dataset=pd.read_csv("Position_Salaries.csv")

#Separating Independent and Dependent Variables
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fitting The Decision Tree Regression To The Dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predciting The Results
y_pred=regressor.predict(X)

#Visualizing The Decision Tree Regression Results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 