import numpy as np
import pandas as pd

from hottbox.core import Tensor, TensorTKD
from hottbox.algorithms.classification import TelVI

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix





class GRTEL:
    def __init__(self, base_clfs, n_classes=1, probability=False, verbose=False):
        self.probability = probability
        self.verbose = verbose
        self.n_classes = n_classes
        self.models = [TelVI(base_clf=base_clfs[i], probability=self.probability, verbose=self.verbose) for i in range(self.n_classes)]
        
    def fit(self, X, y):
        if self.n_classes == 1:
            self.models[0].fit(X, y)
        elif self.n_classes > 1:
            for i in range(self.n_classes):
                print(i, end=" - ")
                self.models[i].fit(X, y[:,i])
            print()
        
    def score(self, X, y):
        if self.n_classes == 1:
            return [self.models[0].score(X, y)]
        elif self.n_classes > 1:
            scores = []
            for i in range(self.n_classes):
                scores.append(self.models[i].score(X, y[:, i]))
            return scores
    
    def grid_search(self, X, y, search_params):
        if self.n_classes == 1:
            self.models[0].grid_search(X, y, search_params)
        elif self.n_classes > 1:
            for i in range(self.n_classes):
                print(i, end=" - ")
                self.models[i].grid_search(X, y[:,i], search_params)
            print()
                
    def predict(self, X):
        predictions = []
        for i in range(self.n_classes):
            predictions.append(self.models[i].predict(X))
        return predictions
    
    def confusion_matrices(self, X, y):
        conf_matrices = []
        predictions = self.predict(X)
        for i in range(self.n_classes):
            conf_matrices.append(confusion_matrix(y[:,i], predictions[i]))
        return conf_matrices





class MultiClassifier:
    def __init__(self, n_classes, verbose=False):
        self.verbose = verbose
        self.n_classes = n_classes
        self.models = [DecisionTreeClassifier() for _ in range(n_classes)]

    def fit(self, X, y):
        for i in range(self.n_classes):
            print(i, end=" - ")
            self.models[i].fit(X, y[:,i])
        print()

    def score(self, X, y):
        scores = []
        for i in range(self.n_classes):
            scores.append(self.models[i].score(X, y[:, i]))
        return scores

    def predict(self, X):
        predictions = []
        for i in range(self.n_classes):
            predictions.append(self.models[i].predict(X))
        return predictions

    def confusion_matrices(self, X, y):
        conf_matrices = []
        predictions = self.predict(X)
        for i in range(self.n_classes):
            conf_matrices.append(confusion_matrix(y[:,i], predictions[i]))
        return conf_matrices





    
