import numpy as np
import sklearn
import sklearn.datasets as ds
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os 

# read the data set
folder = "data"
dataset = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print("Working with {0} images".format(len(dataset)))


# spit the data set
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.20, random_state=42)

# # instantiate learning model (k = 5)
# knn = KNeighborsClassifier(n_neighbors = 5)

# # fitting the model
# knn.fit(X_train, y_train)

# # evaluate the score of the trained classifier on the test dataset
# knc.score(X_test, y_test)