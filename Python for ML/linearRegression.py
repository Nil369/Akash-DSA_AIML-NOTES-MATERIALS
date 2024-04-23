# This Python code snippet is performing the following tasks:
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.DESCR)

diabetes_X  =  diabetes.data[:,np.newaxis,2]

diabetes_X_train  = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

# `diabetes_y_train  = diabetes.target[:-30]` is creating a training set for the target variable in
# the diabetes dataset. It is selecting all the target values except for the last 30 instances in the
# dataset, which will be used for training the linear regression model.
diabetes_y_train  = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# This code snippet is using the `LinearRegression` model from the `linear_model` module in
# scikit-learn to perform linear regression on the diabetes dataset. Here's a breakdown of what each
# line is doing:
model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is:  ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("Weights:  ",  model.coef_)
print("Intercept:  ",  model.intercept_)

plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_predicted)
plt.show()
