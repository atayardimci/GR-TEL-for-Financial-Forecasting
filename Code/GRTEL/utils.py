import numpy as np
import pandas as pd


### Evaluates the confidence in the predicted downturns
def downturn_confidence(actual, predicted):
    n = 0
    x = 0
    for i in range(len(actual)):
        if predicted[i] == 0:
            n += 1
            if predicted[i] == actual[i]:
                x += 1
    
    return None if n == 0 else (n, x, x/n)



# Helper function to display scores of multiclass classification
def print_scores(scores):
    result = []
    for score in scores:
        s = "{:.2f}%".format(score * 100)
        result.append(s)
        
    print('[' + ", ".join(result) + ']')

    

def print_1_percentage(y, n_classes):
    percentages = sum(y == 1.)/len(y)
    percentages = list(percentages) if n_classes > 1 else [percentages]

    print_scores(percentages)



def confusion_matrix_metrics(conf_matrix):
    true_negatives = conf_matrix[0][0]
    false_negatives = conf_matrix[1][0]
    true_positives = conf_matrix[1][1]
    false_positives = conf_matrix[0][1]

    accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    downturn_precision = true_negatives / (true_negatives + false_negatives)

    return accuracy, precision, recall, specificity, downturn_precision

