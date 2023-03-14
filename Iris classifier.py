from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

a=float(input("enter sepal length :"))
b=float(input("enter sepal width :"))
c=float(input("enter petal length :"))
d=float(input("enter petal width :"))
X_new = np.array([[a, b, c, d]])
prediction = knn.predict(X_new)
print("Predicted species name: {}".format(iris_dataset['target_names'][prediction][0]))
