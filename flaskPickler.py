"""
Programmer: Hailey Mueller
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 5/04/21
Bonus?: TBD

Description: This file creates a flask pickler file for our final project.
"""

import pickle # standard python library
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyNaiveBayesClassifier
import mysklearn.myevaluation as myevaluation
import mysklearn.myutils as myutils
import os

# "pickle" an object (AKA object serialization)
# save a Python object to a binary file

# "unpickle" an object (AKA object de-serialization)
# load a Python object from a binary file (back into memory)

# Get data from csv file
table = MyPyTable().load_from_file(os.path.join("input_files","winequality-red.csv"))
y_col = table.get_column("quality", False)
x_cols = table.drop_col("quality")

# Use Naive Bayes to classify
testcase = MyNaiveBayesClassifier()

#Returns x INDEXES
X_train, X_test = myevaluation.stratified_kfold_cross_validation(x_cols,y_col,n_splits=10)
X_train, X_test, y_train, y_test = myutils.getInstances(X_train, X_test, x_cols,y_col)

for i,fold in enumerate(X_train):
    train,test = myutils.normalize_values(X_train[i],X_test[i])
    testcase.fit(train,y_train[i])
    break

packaged_object = testcase
# pickle packaged_object
outfile = open("tree.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()