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

    Author: Hailey Mueller
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

    Author: Hailey Mueller
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

    Author: Hailey Mueller
    """
    return np.round((numWrong)/(numCorrect+numWrong),3)

def group_by(y):
    """Groups y-values based on categories.

    Args:
        y(list): list of y-values

    Returns:
        y_dict(dict): dictionary of values:[indexes]

    Author: Hailey Mueller
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

    Author: Hailey Mueller
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

    Author: Hailey Mueller
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

def get_frequencies(valueList):
    """ Gets the frequencies of the variables within the given list.

    Args:
        valueList(list of obj): list of values being counted

    Returns:
        values(list of obj): list of values in list
        valueSums(list of int): the associated counts of the values
    """
    
    valueSums = []
    values = []
    for value in valueList:
        try:
            index = values.index(value)
            valueSums[index]+=1
        except:
            values.append(value)
            valueSums.append(1)

    return values, valueSums

def select_attribute(instances, available_attributes):
    """ Selects an attribute to partition the tree on next using entropy

    Args:
        current_instances(list of list of obj): list of current instances
        available_attributes(list of str): attribute strings

    Returns:
        str: attribute selected to be split on
    """
    #Define variable(s)
    entropies = []
    #Loop through each attribute --> [att0, att1, att2]
    for attribute in available_attributes:
        #Partition on given attribute, and return dictionary of partitioned values
        partition = partition_instances(instances,attribute,available_attributes)
        entropy = 0
        #Loop through each list in given partition
        for key in partition:
            num_partition_instances = len(partition[key])
            #Calculates frequencies in a partition
            class_columns, frequencies = get_frequencies(create_list(partition[key]))
            #Loop through each frequency in the list
            for frequency in frequencies:
                prob = frequency/num_partition_instances #probability of given frequency occurring
                weight = num_partition_instances/len(instances)
                entropy = entropy + (weight * calculate_entropy(prob)) #sum
        entropies.append(entropy)

    #Determine which attribute has the smallest entropy
    min_entropy = entropies[0]
    min_attr = 0
    #Loop through each entropy value
    for i in range(len(entropies)):
        if entropies[i] < min_entropy:
            min_entropy = entropies[i]
            min_attr = i

    return list(available_attributes.keys())[min_attr]

def create_list(oldList):
    """ Takes a 2D list and takes out one column of values (1D list).

    Args:
        oldList(list of list of obj): 2D list that holds attributes in a given partition.

    Returns:
        newList(list of obj): 1D list of all the calculated values in a partition.
    """

    #If list is empty...
    #return empty list
    if(oldList == []):
        return []

    #Index is assumed to be last value in list
    index = len(oldList[0])-1

    #Create new list
    newList = []
    for value in oldList:
        newList.append(value[index])

    return newList

def calculate_entropy(prob):
    """ Simple function that calculates entropy given a probability

    Args:
        prob(float): a probability value

    Return:
        float: returns the entropy of a given probability
    """
    return -(prob * math.log(prob,2))

def partition_instances(instances, split_attribute, attribute_domains):
    """ This function partitions given instances.

    Args:
        instances(list of list of obj): list of instances
        split_attribute(obj): attribute value to split on
        attribute_domains(dict): dictionary filled with attribute domains

    Return:
        partitions(list of list of obj): list of values that have been partitioned
    """
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is "level"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    # Build a dictionary
    partitions = {} # key (attribute value): value (list of instances with this attribute value)
    # For loop through attributes in dictionary
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            index = int(split_attribute[3:])
            if instance[index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions

def all_same_class(instances):
    """ This functions determines if given instances are all in the same class.

    Args:
        instances(list of list of obj): list of instances

    Return:
        boolean: True if all instance labels matched, False otherwise
    """
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label

def tdidt(current_instances, available_attributes, attribute_domains, previous_instances=None):
    """ This is a recursive function that puts into use the
        TDIDT (ie. Top-Down Induction of Decision Trees) algorithm.

    Args:
        current_instances(list of list of obj): list of current instances
        available_attributes(list of str): attribute strings
        attribute_domains(dict): a dictionary of the attributes and their given values
        previous_instances(list of list of obj): list of instances in the previous partition.

    Return:
        tree(list of obj): returns a list that represents a tree
    """

    # Select an attribute to split on
    available_dict = {}
    for key in attribute_domains:
        try:
            available_attributes.index(key)
            #available
            available_dict[key] = attribute_domains[key]
        except:
            #not available, so don't add
            pass

    if(previous_instances == None):
        previous_instances = len(current_instances)

    split_attribute = select_attribute(current_instances, available_dict)
    available_attributes.remove(split_attribute)
    # cannot split on the same attribute twice in a branch
    # recall: python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains)

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        value_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            #print("CASE 1")
            leaf = ["Leaf",partition[0][len(partition[0])-1],len(partition),len(current_instances)]
            value_subtree.append(leaf)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2")
            values, sums = get_frequencies(create_list(partition))
            max_index = sums.index(max(sums)) #finds index of the max count
            leaf = ["Leaf",values[max_index],max(sums),sum(sums)]
            value_subtree.append(leaf)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            values, sums = get_frequencies(create_list(current_instances))
            max_index = sums.index(max(sums)) #finds index of the max count
            return ["Leaf",values[max_index],len(current_instances),previous_instances]
        else: # all base cases are false... recurse!!
            subtree = tdidt(partition, available_attributes.copy(),attribute_domains, len(current_instances))
            # need to append subtree to value_subtree and appropriately append value subtree
            # to tree
            value_subtree.append(subtree)
        tree.append(value_subtree)
    
    return tree

def classify_tdidt(header,tree,instance):
    """ Takes a decision tree from tdidt and an instance and classifys the instance.

    Args:
        header(list of str): attribute column names
        tree(list of obj): decision tree
        instance(list of obj): list of values used to classify an instance

    Returns:
        str: predicted label for the instance
    """

    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        # Figure out what branch to follow recursively
        for i in range(2,len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match! recurse!
                return tdit_predict(header,value_list[2],instance)
        return 0
    else: #leaf
        return tree[1]

def create_header(numValues):
    """ Creates a basic header for decision trees.

    Args:
        numValues(int): number of values in header

    Returns:
        header(list of str): ex. ["att0","att1",...,"attn"]
    """

    header = []
    for value in range(numValues):
        header.append("att{}".format(value))
    return header

def print_rules(tree, attribute_names, class_name, rule_string="IF"):
    """ A helper function for the Decision Tree Classifier "print_decision_rules."
        Prints out the rules instead of returning a value.

    Args:
        tree(list of obj): a decision tree
        attribute_names(list of str): list of attribute names
        class_name(str): name of the column being predicted
        rule_string(str): decision rule string that is added on with every
            recursive call. Initially contains "IF"
    """

    node_type = tree[0]
    if(node_type == "Attribute"):
        for i in range(2,len(tree)):
            attr_name = attribute_names[int(tree[1][3:])]
            val_name = tree[i][1]
            if(rule_string != "IF"):
                rule_string = rule_string + " AND"
            rule_string = rule_string + " {} == {}".format(attr_name,val_name)
            print_rules(tree[i][2],attribute_names,class_name,rule_string)
            try:
                index = rule_string.rindex(" AND")
                rule_string = rule_string[0:index]
            except:
                rule_string = "IF"
    else:
        rule_string = rule_string + " THEN {} == {}".format(class_name, tree[1])
        print(rule_string)