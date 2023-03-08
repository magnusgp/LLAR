""" Script that embeds the data using the trained autoencoder. """

from autoencoder import VAE

from train import SPECTROGRAMS_PATH, load_fsdd
import os
import numpy as np
import similarity as sim
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score
from tabulate import tabulate
import comparisons as comp
import information_retrieval.metrics as ir

from tqdm import tqdm
from tabulate import tabulate

def embed(vae, spectrograms_path):
    """ Embeds the data using the trained autoencoder. """
    # Instantiate the autoencoder and load the trained weights.
    specs, file_paths = load_fsdd(spectrograms_path)
    embeddings = vae.encoder.predict(specs)
    labels = [int(file_path.split('/')[-1][:3]) for file_path in file_paths]
    labels = np.array(labels)
    
    # save the embeddings and labels
    for i, embedding in enumerate(embeddings):
        tmp = file_paths[i].split('/')
        tmp[2] = 'embeddings'
        np.save('/'.join(tmp), embedding)
        
    np.save('autoencoder/fsdd/embeddings/labels.npy', labels)
    
    return embeddings, labels, file_paths

def comps(embeddings, labels, file_paths, data_path = "autoencoder/fsdd/embeddings/", data_type = "normal", data_name = "none"):
    """Calculates the similarities between every embedding and 
    finds the minimum distance.
    
    Args:
        embeddings (np.array): Array of embeddings
        labels (np.array): Array of labels
    """
    y_true, euc_pred, cos_pred = np.array([]), np.array([]), np.array([])
    
    for i, embedding in tqdm(enumerate(embeddings), total = len(embeddings)):
        euc_filename, _, cos_filename, _ = sim.VAESimilarity(embedding, embeddings, labels, file_paths, i)
        y_true   = np.append(y_true  , int(file_paths[i].split('/')[-1][:3]))
        euc_pred = np.append(euc_pred, int(euc_filename.split('/')[-1][:3]))
        cos_pred = np.append(cos_pred, int(cos_filename.split('/')[-1][:3]))
        
    # get the indices of the misclassified embeddings
    euc_idxs = ((euc_pred == labels) == False).nonzero()
    cos_idxs = ((cos_pred == labels) == False).nonzero()
    
    euc_conf = confusion_matrix(labels, euc_pred, labels = labels)
    cos_conf = confusion_matrix(labels, cos_pred, labels = labels)
    
    euc_acc = accuracy_score(labels, euc_pred)
    cos_acc = accuracy_score(labels, cos_pred)
    
    np.save('autoencoder/fsdd/comparisons/' + 'euc_acc.npy', euc_acc)
    np.save('autoencoder/fsdd/comparisons/' + 'cos_acc.npy', cos_acc)
    np.save('autoencoder/fsdd/comparisons/' + 'eud_pred.npy', euc_pred)
    np.save('autoencoder/fsdd/comparisons/' + 'cos_pred.npy', cos_pred)
    np.save('autoencoder/fsdd/comparisons/' + 'y_true.npy', y_true)
    
    return euc_acc, cos_acc, euc_pred, cos_pred, y_true
    
if __name__ == "__main__":
    vae = VAE.load("autoencoder/runs/run4")
    embeddings, labels, file_paths = embed(vae, SPECTROGRAMS_PATH)
    try:
        euc_acc = np.load('autoencoder/fsdd/comparisons/' + 'euc_acc.npy')
        cos_acc = np.load('autoencoder/fsdd/comparisons/' + 'cos_acc.npy')
        euc_pred = np.load('autoencoder/fsdd/comparisons/' + 'eud_pred.npy')
        cos_pred = np.load('autoencoder/fsdd/comparisons/' + 'cos_pred.npy')
        y_true = np.load('autoencoder/fsdd/comparisons/' + 'y_true.npy')
        print(f"\nFound saved comparison metrics.")
        
    except:
        euc_acc, cos_acc, euc_pred, cos_pred, y_true = comps(embeddings, labels, file_paths)
    
    # IR metrics
    k = 5
    print("Information Retrieval Table for trained VAE model:\n")
    print(tabulate([['Euclidean Distance', euc_acc, ir.Precision(labels, euc_pred, y_true), ir.Recall(labels, euc_pred, y_true), 
                        ir.F1(labels, euc_pred, y_true), ir.MeanAveragePrecision(labels, euc_pred, y_true), ir.AccuracyAtK(labels, euc_pred, y_true, k)], 
                    ['Cosine Distance', cos_acc, ir.Precision(labels, cos_pred, y_true), ir.Recall(labels, cos_pred, y_true), 
                        ir.F1(labels, cos_pred, y_true), ir.MeanAveragePrecision(labels, cos_pred, y_true), ir.AccuracyAtK(labels, cos_pred, y_true, k)]], 
                    headers=['IR Type', 'Accuracy', 'Precision', 'Recall', 
                            'F1-Score', 'MAP', 'Accuracy @ K (K = 5)']))
    
    
    