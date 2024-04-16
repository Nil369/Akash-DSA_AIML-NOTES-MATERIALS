# Loading necessary modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Datasets
iris = datasets.load_iris()

# Printing Descriptions & Features
# print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0],labels[0])

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)
preds = clf.predict([[31,1,1,1]])
print(preds)
