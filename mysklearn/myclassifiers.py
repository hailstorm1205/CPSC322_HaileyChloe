'''
Programmer: Hailey Mueller & Chloe Crawford
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 5/03/21
Bonus?: TBD

Description: This file holds the classifier classes used for our final project. Those classifiers
    include the Random Forest, Naive Bayes, and Decision Tree classifiers.
'''

import mysklearn.myutils as myutils
import numpy as np
import copy

# Random Forest Classifer
class MyRandomForestClassifier:
    """ Represents a simple Random Forest classifier.
    Attributes:
        N(int): Number of trees to be generated
        M(int): Subset of "better" trees to be used to make decisions
        trees(list of MyDecisionTreeClassifier obj): List of all of the decision trees that 
            make up the forest
        X_train(list of list of obj): The list of training instances (samples). 
            The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    Notes:
        I decided to make a list of decision tree objects because that already covers a lot of the functionality
            that we will need. The only problem is using compute_random_subset() right before the call to 
            select_attribute() in tdidt. Obviously we don't want to use that methodology for the first 
            decision tree classifier test, so we'll have to figure that out
    """
    def __init__(self, N=100, M=30):
        self.N = N
        self.M = M
        self.trees = None
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train, X_test, y_test):
        """Fits many decision tree classifiers to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            X_test(list of list of obj): The list of testing instances, used to determine accuracy of trees
            y_test(list of obj): The target y values (parallel to X_test), used to determine accuracy of trees

        Notes:
            
        """
        # Call train_test_split to get "test" and "remainder" data (happens before this function is called)
        # Loop N times
        #   Call compute_bootstrapped_sample to get a subset of the "remainder" data
        #   Call tdidt with the bootstrapped sample to build a decision tree
        # Test performance of all trees to get the M best ones
        # Use majority voting to make predictions (predict method)
        self.X_train = X_train
        self.y_train = y_train
        N_trees = []
        for _ in range(self.N):
            new_X_train, new_y_train = myutils.compute_bootstrapped_sample(X_train, y_train)
            tree = MyDecisionTreeClassifier()
            tree.fit(new_X_train, new_y_train, is_forest=True)
            N_trees.append(tree)

        self.trees = []
        # Test tree performance
        #   for the M best trees, append them to self.trees
        accuracy_list = []
        for tree in N_trees:
            accuracy = 0
            predict_list = tree.predict(X_test)
            for i in range(len(predict_list)):
                if predict_list[i] == y_test[i]:
                    accuracy += 1
            accuracy_list.append(myutils.calculateAccuracy(accuracy, len(y_test) - accuracy))
        prev_best = 1.1
        M_indexes = np.argpartition(accuracy_list, -self.M)[-self.M:]
        print(accuracy_list)
        print(M_indexes)
        for index in M_indexes:
            self.trees.append(N_trees[index])
        

    def predict(self):
        pass

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

    def fit(self, X_train, y_train, is_forest=False):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            is_forest(bool): determines whether this fit was called by MyRandomForestClassifier

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
        self.tree = myutils.tdidt(train, available_attributes, attribute_domains, is_forest)
        
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

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Determine self.priors

        # category_indexes = {cat1: [val1,...valn], ..., catn: [val1,...,val2]}
        category_indexes = {}
        for i, row in enumerate(X_train):
            try:
                category_indexes[y_train[i]]
                #Value in dictionary
                category_indexes[y_train[i]].append(i)
            except:
                #Value not in dictionary
                category_indexes[y_train[i]] = [i]

        # self.priors = {cat1: val, ..., catn: val}
        self.priors = {}
        for key, values in category_indexes.items():
            self.priors[key] = len(values)/len(X_train)

        # Determine self.posteriors

        #Initialize y-category posteriors
        self.posteriors = {}
        for index, key in enumerate(self.priors):
            self.posteriors[key] = {}

        for i in range(len(self.X_train[0])):
            self.posteriors[key][i] = {}

        #This for loop is necessary to make sure all categories
        #are being added to posteriors, even if probability = 0
        posterior_categories = [] #2d array
        for row in self.X_train:
            for i, value in enumerate(row):
                try:
                    #Value already in array
                    posterior_categories[i].index(value)
                except:
                    try:
                        #Add value to list
                        posterior_categories[i].append(value)
                    except:
                        #Need to make new list
                        posterior_categories.append([value])

        #Sorts rows that have int values for testing
        for row in posterior_categories:
            if(type(row[0]) is int):
                row.sort()

        # self.posteriors = {cat1: {0: {0: val, ..., n: val}, ..., n: {...}}, ..., catn: {{...}}}
        for key, values in category_indexes.items():
            for j in range(len(self.X_train[0])):
                # j represents the index of the attribute being used
                self.posteriors[key][j] = {} 
                newList = []
                for index in values:
                    # index represents the indexes of the 
                    # rows in the given category
                    newList.append(X_train[index][j])

                counts = {}
                #Initialize counts
                for value in posterior_categories[j]:
                    counts[value] = 0
                
                for value in newList:
                    counts[value] += 1

                for k,value in counts.items():
                    counts[k] = value/len(newList)

                self.posteriors[key][j] = counts

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        #Initialize Probabilities 2D Dictionary
        #that will store all the probabilities
        probabilities = {}
        for test in range(len(X_test)):
            probabilities["Test: {}".format(test+1)] = {}

        #Loop through posteriors and add probabilities to probabilities
        #dictionary accordingly
        for x in range(len(X_test)): # in case there are multiple test cases
            for i in self.posteriors: #category (ex. yes/no)
                probabilities["Test: {}".format(x+1)][i] = []
                for j in self.posteriors[i]: #column (ex. standing, job_status)
                    try:
                        probabilities["Test: {}".format(x+1)][i].append(self.posteriors[i][j][X_test[x][j]])
                    except:
                        probabilities["Test: {}".format(x+1)][i].append(0)

        #Determine which Naive Bayes probability is greater
        #in order to predict y value
        y_predicted = []
        index = 0
        for i in probabilities: #each test case
            prevProbability = 0.0
            for j in probabilities[i]: #each category
                probability = 1.0
                for value in probabilities[i][j]: #list
                    probability*=value
                probability*=self.priors[j]
                try:
                    y_predicted[index]
                    if(prevProbability < probability):
                        y_predicted[index] = j
                        prevProbability = probability
                except:
                    y_predicted.append(j)
                    prevProbability = probability
            index+=1

        return y_predicted