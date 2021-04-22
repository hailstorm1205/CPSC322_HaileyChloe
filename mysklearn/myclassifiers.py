'''
Programmer: Hailey Mueller & Chloe Crawford
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 4/22/21
Bonus?: TBD

Description: This file holds the classifier classes used for our final project. Those classifiers
    include the Random Forest, kNN, and Decision Tree classifiers.
'''

import mysklearn.myutils as myutils
import copy

# Random Forest Classifer
class MyRandomForestClassifier:
    """ Represents a simple Random Forest classifier.
    Attributes:
        tbd
    Notes:
        tbd
    """
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

# Taken from Chloe's Code
class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        distances = []
        neighbor_indices = []

        for i in range(len(X_test) - 1):
            distance_temp = []
            neighbor_temp = []
            for j in range(len(X_test[i])):
                distance_temp.append(utils.calc_distance(self.X_train[i][j], self.X_train[i + 1][j], X_test[i][j], X_test[i + 1][j]))
                neighbor_temp.append(i)
            distances.append(distance_temp)
            neighbor_indices.append(neighbor_temp)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []

        indices = []
        distances, neighbor_indices = self.kneighbors(X_test)
        #distances.sort()
        for i in range(self.n_neighbors):
            min_val = distances[0]
            min_index = neighbor_indices[0]
            for j in range(len(distances)):
                if distances[j] < min_val:
                    if len(indices) > 0 and distances[j] > distances[indices[-1]]:
                        min_val = distances[j]
                        min_index = neighbor_indices[j]
                    elif len(indices) == 0:
                        min_val = distances[j]
                        min_index = neighbor_indices[j]
            indices.append(min_index)

        count = 0
        val1 = self.y_train[0]
        val2 = ""
        for value in self.y_train:
            if value != val1:
                val2 = value
                break
            
        for val in indices:
            if self.y_train[0] == val:
                count += 1
        
        if count > self.n_neighbors // 2:
            y_predicted.append(val1)
        else:
            y_predicted.append(val2)


        return y_predicted

# Taken from Hailey's Code
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # Define variables
        self.X_train = X_train
        self.y_train = y_train

        # Compute a "header" ["att0", "att1", ...]
        header = []
        for i,value in enumerate(X_train[0]):
            header.append("att{}".format(i))

        #print(header)

        # Compute the attribute domains dictionary
        attribute_domains = {}
        for i in range(len(header)):
            attribute_domains[header[i]] = []
            for j in range(len(X_train)):
                #print(X_train[j][i])
                try:
                    attribute_domains[header[i]].index(X_train[j][i])
                    #found
                except:
                    #not found
                    attribute_domains[header[i]].append(X_train[j][i])
            attribute_domains[header[i]].sort()

        #print(attribute_domains)

        # my advice is to stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # initial call to tdidt current instances is the whole table (train)
        available_attributes = header.copy() # python is pass object reference
        self.tree = myutils.tdidt(train, available_attributes, attribute_domains)
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = myutils.create_header(len(self.X_train[0]))
        try:
            predictions = []
            for test in X_test:
                predict = myutils.classify_tdidt(header,self.tree,test)
                predictions.append(predict)
            return predictions
        except:
            return None

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if(attribute_names is None):
            attribute_names = myutils.create_header(len(self.X_train[0]))

        myutils.print_rules(self.tree,attribute_names,class_name)