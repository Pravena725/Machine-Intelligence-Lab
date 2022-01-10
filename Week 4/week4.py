import numpy as np
import math

class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # TODO
        try:
            dist_matrix = []
            pval=self.p
            data_to_be_tested=self.data
            sum=0.0
       
            for i in x: #the data that is provided
                row=[] #the row of the matrix
                for j in data_to_be_tested: #the part that we have to check for
                    #formula = pow(np.sum(abs(given_data-test_data)**pval),1/pval
                    add = np.sum(abs(i-j)**pval)
                    row.append(pow(add,1/np.float64(pval)))
                dist_matrix.append(row)
            return dist_matrix
        
        except:
            pass
                
        

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        # TODO
        try:
            neigh_dists_find = []
            idx_of_neigh_find = []
        
            for i in self.find_distance(x):
                dist_sort = list(np.sort(i)[0:self.k_neigh]) #sorted dist of inputs closest to k closest neighbours
                index_neigh_sort = list(np.argsort(i)[0:self.k_neigh]) #indirect sorting of kNN 
            
                neigh_dists_find.append(dist_sort)            
                idx_of_neigh_find.append(index_neigh_sort)
            
            return [neigh_dists_find,idx_of_neigh_find]
            
        except:
            pass

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        pred_find = []
        knn_tar = self.k_neighbours(x)[1]
        for value in knn_tar:
            target_count = np.bincount(self.target[value]) #count no. of times of each value in array of +ve ints
            pred_find.append(target_count.argmax())
            
        return pred_find
        pass

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        # TODO
        evaluated_values = self.predict(x)
        numerator = np.sum(evaluated_values == y)
        accuracy_find = numerator *100/len(y)
        return accuracy_find
          
        pass
