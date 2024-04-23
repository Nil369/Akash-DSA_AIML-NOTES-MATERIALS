# This code snippet is training a logistic regression classifier using the Iris dataset from
# scikit-learn. The goal is to predict whether a flower is Iris Virginica or not based on the petal
# width feature.
# Train a logistic regression classifier to predict whether a flower is iris virginica or not
# This code snippet is training a logistic regression classifier using the Iris dataset from
# scikit-learn. Here's a breakdown of what the code is doing:
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(list(iris.keys()));
# print(iris['data'].shape);
# print(iris['target']);
# print(iris['DESCR']);

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)


# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X,y)
example = clf.predict(([[2.6]]))
print(example)

# Using matplotlib to plot the visualization
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)
print(y_prob)
plt.plot(X_new, y_prob[:,1], "g-", label="virginica")
# `plt.show()` is a function in matplotlib that displays all figures created using matplotlib. When
# you call `plt.show()`, it will open a window displaying the plot that you have created using
# matplotlib functions like `plt.plot()`. This function is typically used at the end of your plotting
# code to show the final visualization to the user.
plt.show()

# print(y);
# print(iris["data"]);
# print(X);

