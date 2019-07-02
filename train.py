import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os 
from glob import glob

def processImg():
   # create a label dict to map strings to numerals
   label_dic = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
             "six": 6, "seven": 7, "eight": 8, "nine": 9}

   feat = []
   labels = []

   # find all jpg in data folder
   folder = glob("data/*.jpg")
   
   # loop through each file and retrieve values + label
   for i, file in enumerate(folder):
      new_label = os.path.split(file)[1].split("_")[0] 
      labels.append(label_dic[new_label])

      image = cv2.imread(file)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      gray = gray.reshape(400)
      feat.append(gray)

   # format as array and return
   feat = np.asarray(feat).astype(np.float32)
   labels = np.asarray(labels).astype(np.float32)

   return feat, labels

# generate features and labels
features, labels = processImg()

# split the training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20)
# perform knn 
knn = KNeighborsClassifier(n_neighbors = 5)
# fit the knn using training data
knn.fit(X_train, y_train)
# predict on new dataset
y_pred = knn.predict(X_test)
# print accuracy score
print(accuracy_score(y_test, y_pred))

# create a confusion matrix
con_matrix = confusion_matrix(y_test, y_pred)
print(con_matrix)


# print recall and precision
for i in range(0, 10):
   # recall = TP / (TP + FN)
   tp = con_matrix[i][i]
   fn_tp = con_matrix[i, :].sum()
   recall = tp / fn_tp

   # precision = TP / (TP + FP)
   tp = con_matrix[i][i]
   fp_tp = con_matrix[:, 1].sum()
   precision = tp / fp_tp

   print("-------Class", i, "Recall-------")
   print("True Positives:                                 ", tp)
   print("True Positives & False Negatives:               ", fn_tp)
   print("Recall:                                          {:.3f}%".format(recall*100), end="\n")

   print("-------Class", i, "Precision-------")
   print("True Positives:                                 ", tp)
   print("True Positives & False Negatives:               ", fp_tp)
   print("Precision:                                       {:.3f}%".format(precision*100), end="\n\n")