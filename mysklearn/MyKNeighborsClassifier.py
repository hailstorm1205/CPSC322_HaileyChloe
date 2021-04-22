import mysklearn.myutils as utils

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