"""Script that inplements information retrieval metrics for the comparisons."""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

class Precision:
    """Precision metric for the comparisons."""
    def __init__(self, labels, y_true, pred):
        self.labels = labels
        self.pred = pred
        self.y_true = y_true
        self.conf = confusion_matrix(self.y_true, self.pred)
        try:
            self.precision = np.diag(self.conf) / np.sum(self.conf, axis = 0)
        except:
            self.precision = np.array([0])
        self.mean_precision = np.mean(self.precision)
        
    def __repr__(self):
        return str(self.mean_precision.round(3))
    
class Recall:
    """Recall metric for the comparisons."""
    def __init__(self, labels, y_true, pred):
        self.labels = labels
        self.pred = pred
        self.y_true = y_true
        self.conf = confusion_matrix(self.y_true, self.pred)
        try:
            self.recall = np.diag(self.conf) / np.sum(self.conf, axis = 1)
        except:
            self.recall = np.array([0])
        self.mean_recall = np.mean(self.recall)
        
    def __repr__(self):
        return str(self.mean_recall.round(3))
    
class F1:
    """F1 metric for the comparisons."""
    def __init__(self, labels, y_true, pred):
        self.labels = labels
        self.pred = pred
        self.y_true = y_true
        self.conf = confusion_matrix(self.y_true, self.pred)
        self.precision = np.diag(self.conf) / np.sum(self.conf, axis = 0)
        self.recall = np.diag(self.conf) / np.sum(self.conf, axis = 1)
        
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        # replace nan with 0
        self.f1[np.isnan(self.f1)] = 0
        self.mean_f1 = np.mean(self.f1)
        
    def __repr__(self):
        return str(self.mean_f1.round(3))
    
class Accuracy:
    """Accuracy metric for the comparisons."""
    def __init__(self, labels, y_true, pred):
        self.labels = labels
        self.pred = pred
        self.y_true = y_true
        self.accuracy = accuracy_score(self.y_true, self.pred)
        
    def __repr__(self):
        return str(self.accuracy.round(3))
    
class MeanAveragePrecision:
    """Mean Average Precision metric for the comparisons."""
    def __init__(self, labels, y_true, top_k_preds, k):
        # self.labels = labels
        # self.pred = pred
        # self.y_true = y_true
        # self.conf = confusion_matrix(self.y_true, self.pred, labels = np.unique(self.labels))
        # self.precision = np.diag(self.conf) / np.sum(self.conf, axis = 0)
        # self.mean_precision = np.mean(self.precision)
        
        self.y_true = y_true
        self.y_pred = top_k_preds
        self.k = k
        
        self.map = self.mean_average_precision(self.y_true, self.y_pred, self.k)
        

    def mean_average_precision(self, y_true, y_pred, k=5):
        """
        Calculate the mean average precision for a set of queries.

        Arguments:
        - y_true: true labels for each query (list or numpy array)
        - y_pred: predicted labels for each query, sorted by relevance (2D numpy array)
        - k: number of predictions to consider for each query (integer)

        Returns:
        - mean average precision (float)
        """

        aps = []

        for i, query_true in enumerate(y_true):
            query_pred = y_pred[i][:k]

            # calculate precision at k for this query
            num_correct = 0
            precision_at_k = []
            for j, pred_label in enumerate(query_pred):
                if pred_label == query_true:
                    num_correct += 1
                    precision_at_k.append(num_correct / (j+1))
            if not precision_at_k:
                continue
            ap = np.mean(precision_at_k)
            aps.append(ap)

        return np.mean(aps)


    def __repr__(self):
        return str(self.map.round(3))
    
class AccuracyAtK:
    """Accuracy at K metric for the comparisons."""
    # def __init__(self, labels, y_true, pred, k):
    #     self.labels = labels
    #     self.pred = pred
    #     self.y_true = y_true
    #     self.k = k
    #     self.conf = confusion_matrix(self.y_true, self.pred, labels = np.unique(self.labels))
    #     self.accuracy_at_k = np.diag(self.conf) / np.sum(self.conf, axis = 1)
    #     self.mean_accuracy_at_k = np.mean(self.accuracy_at_k)
    
    def __init__(self, labels, y_true, top_k_preds, k):
        self.labels = labels
        self.top_k_preds = top_k_preds
        self.y_true = y_true
        self.k = k
        self.accuracy_at_k = np.array([1 if self.y_true[i] in self.top_k_preds[i][:self.k] else 0 for i in range(len(self.y_true))])
        self.mean_accuracy_at_k = np.mean(self.accuracy_at_k)
        
    def __repr__(self):
        return str(self.mean_accuracy_at_k.round(3))
    
if __name__ == "__main__":
    pass