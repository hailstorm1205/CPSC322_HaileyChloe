'''
Programmer: Hailey Mueller
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 4/22/21
    
Description: This file contains the functions that splits the dataset based on the given parameters.
    The algorithms are train test split, k-fold cross validation, and stratified k-fold cross validation.
    This file also holds the function that creates a confusion matrix for the given data.
'''

import numpy as np
import copy
import mysklearn.myutils as myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of taget y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    copyX = copy.deepcopy(X)
    copyY = copy.deepcopy(y)
    if random_state is not None:
        # TODO: seed your random number generator
        #Seed random number generator
        np.random.seed(random_state)
    
    if shuffle: 
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        copyX, copyY = myutils.randomize_in_place(copyX,copyY)

    #Define Variables
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    prop_sum = 0.0
    numTest = 0
    proportion = 1.0/float(len(X))

    #Determine how many values to put in test set
    while(prop_sum < test_size):
        numTest = numTest + 1
        prop_sum = prop_sum + proportion
    
    #Put values in train/test sets
    for i in range(len(X)):
        if(test_size>=1):
            if(i<=len(X)-1-test_size):
                X_train.append(copyX[i])
                y_train.append(copyY[i])
            else:
                X_test.append(copyX[i])
                y_test.append(copyY[i])
        else:
            if(i<=len(X)-1-numTest):
                X_train.append(copyX[i])
                y_train.append(copyY[i])
            else:
                X_test.append(copyX[i])
                y_test.append(copyY[i])

    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    #Define variables
    X_train_folds = []
    X_test_folds = []
    indexes = list(range(len(X)))
    index = 0

    #Create folds
    for i in range(n_splits):
        test = []
        train = []
        #Determine how many to put in test
        if((len(X) % n_splits) > i):
            numTest = len(X) // n_splits +1
        else:
            numTest = len(X) // n_splits
        for j in range(numTest):
            if(index < len(X)):
                test.append(index)
                indexes.pop(indexes.index(index))
                index = index + 1
        for index1 in indexes:
            train.append(index1)
        X_test_folds.append(test)
        X_train_folds.append(train)
        indexes = list(range(len(X)))

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """

    #Define variables
    X_train_folds = []
    X_test_folds = []

    #Create dictionary    
    y_dict = myutils.group_by(y)

    #Split data
    folds = [[] for _ in range(n_splits)]
    for category in y_dict.keys():
        index = y_dict[category]
        for i in range(len(index)):
            folds[i % n_splits].append(index[i])

    #Add data to train and testing sets
    for i in range(n_splits):
        train = []
        for j in range(n_splits):
            if i != j:
                for item in folds[j]:
                    train.append(item)
        test = folds[i]
        X_train_folds.append(train)
        X_test_folds.append(test)
    
    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    #Define variables
    matrix = []
    #Creates matrix dimensions
    for i in range(len(labels)):
        matrix.append([])
        for j in range(len(labels)):
            matrix[i].append(0)

    for i in range(len(y_true)):
        trueIndex = -1
        predIndex = -1
        #Get indexes of true and predicted values
        for j, label in enumerate(labels):
            if(label == y_true[i]):
                trueIndex = j
            if(label == y_pred[i]):
                predIndex = j
        matrix[trueIndex][predIndex] = matrix[trueIndex][predIndex] + 1

    return matrix
    