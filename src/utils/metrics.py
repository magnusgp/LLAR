"""Script that inplements information retrieval metrics for the comparisons."""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

class Precision:
    """Precision metric for the comparisons."""
    def __init__(self, labels, y_true, pred):
        self.labels = labels
        self.pred = pred
        self.y_true = y_true
        self.conf = confusion_matrix(self.y_true, self.pred, labels = np.unique(self.labels))
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
        self.conf = confusion_matrix(self.y_true, self.pred, labels = np.unique(self.labels))
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
        self.conf = confusion_matrix(self.y_true, self.pred, labels = np.unique(self.labels))
        self.precision = np.diag(self.conf) / np.sum(self.conf, axis = 0)
        self.recall = np.diag(self.conf) / np.sum(self.conf, axis = 1)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
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
    def __init__(self, labels, y_true, pred):
        self.labels = labels
        self.pred = pred
        self.y_true = y_true
        self.conf = confusion_matrix(self.y_true, self.pred, labels = np.unique(self.labels))
        self.precision = np.diag(self.conf) / np.sum(self.conf, axis = 0)
        self.mean_precision = np.mean(self.precision)
        
    def __repr__(self):
        return str(self.mean_precision.round(3))
    
class AccuracyAtK:
    """Accuracy at K metric for the comparisons."""
    def __init__(self, labels, y_true, pred, k):
        self.labels = labels
        self.pred = pred
        self.y_true = y_true
        self.k = k
        self.conf = confusion_matrix(self.y_true, self.pred, labels = np.unique(self.labels))
        self.accuracy_at_k = np.diag(self.conf) / np.sum(self.conf, axis = 1)
        self.mean_accuracy_at_k = np.mean(self.accuracy_at_k)
        
    def __repr__(self):
        return str(self.mean_accuracy_at_k.round(3))