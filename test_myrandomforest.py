'''
Programmer: Hailey Mueller & Chloe Crawford
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 5/04/21
Bonus?: TBD

Description: This file is a testing function for the MyRandomForestClassifier object
'''

import numpy as np
import mysklearn.myutils as myutils
from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable

def test_random_forest_fit():
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
    myutils.prepend_attribute_label(interview_table, interview_header)

    interview_pytable = MyPyTable(column_names=interview_header, data=interview_table)
    y_col = interview_pytable.get_column("interviewed_well", False)
    x_cols = interview_pytable.drop_col("interviewed_well")

    many_trees = MyRandomForestClassifier()
    X_sample, y_sample = myutils.compute_bootstrapped_sample(x_cols, y_col)
    X_train, X_test, y_train, y_test = myutils.train_test_split(X_sample, y_sample, .33)
    many_trees.fit(X_train, y_train, X_test, y_test)
    y_predicted = many_trees.predict(X_test)

    numCorrectPredictions = 0
    numWrongPredictions = 0
    for i in range(len(y_test)):
        values = [y_predicted[i], y_test[i]] #predicted/actual
        if(values[0]==values[1]):
            numCorrectPredictions = numCorrectPredictions+1
        else:
            numWrongPredictions = numWrongPredictions+1

    accuracy = np.round((numCorrectPredictions)/(numCorrectPredictions+numWrongPredictions),3)
    error_rate = np.round((numWrongPredictions)/(numCorrectPredictions+numWrongPredictions),3)

    print("-----------------------------------------------------------")
    print("Accuracy and Error Rate")
    print("-----------------------------------------------------------")
    print()
    print("Random Forest: accuracy = {}, error rate = {}".format(accuracy,error_rate))
    print()
    print("Because of the random aspect of this classifier, this will not always pass the tests")
    print()
    print("Predicted table: " + str(y_predicted))
    print("Testing set:     " + str(y_test))
    for i in range(len(y_test)):
        assert y_predicted[i] == y_test[i]