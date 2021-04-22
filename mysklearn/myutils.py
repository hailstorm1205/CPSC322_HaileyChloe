'''
Programmer: Chloe Crawford & Hailey Mueller
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 4/22/21
    
Description: This file contains various utility functions used for our classifiers.
'''

import math
import csv
import copy

# Splits dataset into test and train data
def train_test_split(X, y, test_size):
    num_instances = len(X) 
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size)
    split_index = num_instances - test_size 

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Calculates distance between two points
def calc_distance(x1, x2, test1, test2):
    return math.sqrt(((x1 - test1) ** 2) + ((x2 - test2) ** 2))

def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "N/A" or row[col_index].count > 0:
            col.append(row[col_index])
    return col

# Converts strings to numerical values
def convert_to_numeric(values):
    for i in range(len(values)):
        for j in range(len(values[i])):
            numeric_value = float(values[i][j])
            values[i][j] = numeric_value
    return values

# Reads a table from a CSV file
def read_table(filename):
    infile = open(filename, "r")
    the_reader = csv.reader(infile, dialect='excel')
    table = []
    for row in the_reader:
        if len(row) > 0:
            table.append(row)
    infile.close()
    header = table[0]
    del table[0]
    return table, header

# Pretty prints a 2D array
def pretty_print(array):
    for val in array:
        for item in val:
            print(item, end=" ")
        print()

def getInstances(X_train,X_test,x_values,y_values):
    """Gets instances from indexes in X_train and X_test from x_values.
    
    Args:
        X_train(list of list of int): list of X_train's with their given indexes
            ex. [[1,2,3],[4,5,6]]
        X_test(list of list of int): list of X_test's with their given indexes
            ex. [[4,5,6],[1,2,3]]
        x_values(list of list of obj): list of x-values separated by instance
            ex. [["Greg",500,2],["Hailey",250,3]] (X_train[0][1] should refer to the "Hailey" instance)
    
    Returns:
        newTrain(list of list of list of obj): 3D list that replaces indexes with x-values
        newTest(list of list of list of obj): 3D list that replaces indexes with x-values
    """

    newTrain = []
    yTrain = []
    for i, train in enumerate(X_train): #loop through each train
        newTrain.append([])
        yTrain.append([])
        for index in train: #loop through each index
            newTrain[i].append(x_values[index])
            yTrain[i].append(y_values[index])

    newTest = []
    yTest = []
    for i, test in enumerate(X_test): #loop through each train
        newTest.append([])
        yTest.append([])
        for index in test: #loop through each index
            newTest[i].append(x_values[index])
            yTest[i].append(y_values[index])

    return newTrain, newTest, yTrain, yTest

def calculateAccuracy(numCorrect, numWrong):
    """ Simple function that calculates the accuracy given
        the number of correct/incorrect answers.

    Args:
        numCorrect(int): the number of correct values
        numWrong(int): the number of incorrect values

    Returns:
        float: returns the accuracy rounded to 3 decimals
    """
    return np.round((numCorrect)/(numCorrect+numWrong),3)

def calculateErrorRate(numCorrect, numWrong):
    """ Simple function that calculates the error rate given
        the number of correct/incorrect answers.

    Args:
        numCorrect(int): the number of correct values
        numWrong(int): the number of incorrect values

    Returns:
        float: returns the error rate rounded to 3 decimals
    """
    return np.round((numWrong)/(numCorrect+numWrong),3)

def group_by(y):
    """Groups y-values based on categories.

        Args:
            y(list): list of y-values

        Returns:
            y_dict(dict): dictionary of values:[indexes]
    """

    y_dict = {}
    for i, value in enumerate(y):
        try:
            y_dict[value]
            #Value in dictionary
            y_dict[value].append(i)
        except:
            #Value not in dictionary
            y_dict[value] = [i]

    return y_dict

def normalize_values(train, test=None):
    """Normalizes values on a scale from 0 to 1.

        Args:
            train(list of list of obj): training set
            test(list of list of obj): testing set

        Returns:
            newTrain(list of list of obj): the normalized training set
            newTest(list of list of obj): the normalized testing set that is only
                returned if the test parameter is passed in.
    """

    #Define variables
    newTrain = copy.deepcopy(train)
    coordLists = []

    #Find max amount of coordinates
    maxCoords = 0
    for row in train:
        if(len(row) > maxCoords):
            maxCoords = len(row)

    #Create a list for each coordinate in train
    #if train = [[0,1,2],[3,4,5],...,[n,n,n]]
    #then coordLists = [[1],[2],[3]]
    for i in range(maxCoords):
        coordLists.append([])

    #Add coordinates to respective rows
    for i,row in enumerate(newTrain):
        for j,coord in enumerate(row):
            coordLists[j].append(coord)
        
    #Normalize Train
    for i in range(len(newTrain)):
        for j in range(len(newTrain[i])):
            val = (newTrain[i][j] - min(coordLists[j])) / ((max(coordLists[j])-min(coordLists[j]))*1.0)
            newTrain[i][j] = val

    #Normalize Test
    if test is not None:
        newTest = copy.deepcopy(test)
        for i in range(len(newTest)):
            for j in range(len(newTest[i])):
                val = (newTest[i][j] - min(coordLists[j])) / ((max(coordLists[j])-min(coordLists[j]))*1.0)
                newTest[i][j] = val
        
        return newTrain, newTest

    return newTrain

def removePercentage(column):
    """If a value within the given column has a percent sign, that percent sign will be removed
        and then the new int value will be multipled by 0.01 to make it a proper fraction decimal.

        Args:
            column(list): list of data used to get information from
            
        Returns:
            list: altered "column" list with removed percent signs
    """
    
    #Define variables
    newColumn = []
    
    #Add new values to newColumn
    for val in column:
        try:
            index = val.index("%")
            newVal = round(int(val[0:index])*.01,2)
            newColumn.append(newVal)
        except:
            #No percent sign was found
            newColumn.append(val)
        
    return newColumn