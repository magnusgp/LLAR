import numpy as np
import os
from tqdm import tqdm
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score
import information_retrieval.metrics as ir

import similarity as sim

import faiss


def comparisons(data_path='output/embeddings/ECS50/', data_type = "normal", data_name = "mean"):
    """Function that compares all embeddings to each other and plots a matrix that shows how often
    the most similar embeddings beling to the same class/folder as the original embedding.

    Args:
        data_path (str, optional): Path of the embeddings. Defaults to 'output/embeddings/'.
    """
    y_true, pred = np.array([]), np.array([])
    labels = [int(folder[:3]) for folder in os.listdir(data_path) if folder != '.DS_Store']
    
    embeddings = []
    # get all the embeddings
    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
        embeddings.append(['output/embeddings/ECS50/' + folder + '/' + file for file in os.listdir('output/embeddings/ECS50/' + folder)])
        
    # flatten the list
    embeddings = [item for sublist in embeddings for item in sublist if item.endswith('.npy') and not item.endswith('_trimmed.npy')]
    
    for i, folder in tqdm(enumerate(os.listdir(data_path)), total = len(os.listdir(data_path))):
        if folder == '.DS_Store':
            continue
        for file in os.listdir(data_path + folder):
            if file.endswith('.npy') and not file.endswith('_trimmed.npy'):
                # load the current embedding
                input_vector = str(data_path + folder + '/' + file)
                y_true = np.append(y_true, int(folder[:3]))
                # compare the current embedding to all other embeddings
                input_name = input_vector.split('/')[-1].split('.')[0]
                input_full_name = input_vector
                input_vector = np.load(input_vector)
                inputs = []
                if "mean" in data:
                    inputs.append(np.mean(input_vector, axis=0))
                if "var" in data:
                    inputs.append(np.var(input_vector, axis=0))
                if "delta" in data:
                    inputs.append(np.mean(librosa.feature.delta(input_vector),axis=0))
                    
                # meaninput = np.mean(input_vector, axis=0)
                # varinput = np.var(input_vector, axis=0)
                # deltainput = np.mean(librosa.feature.delta(input_vector),axis=0)
                
                # input_vector = np.concatenate((meaninput, varinput, deltainput), axis=0)
                input_vector = np.concatenate(inputs, axis=0)
                embedded_vectors = {}
                # load the embedded vectors from the folder into a dictionary with their filenames as keys
                for i, file_name in enumerate(embeddings):
                    # skip the input vector
                    if input_name in file_name:
                        continue
                    else:
                        embedded_vectors[file_name] = np.load(file_name)
                
                eucdistances, cosdistances = {}, {}
                shapes = {}
                count = 0
                # calculate the euclidean distance between the input vector and the embedded vectors
                for file_name, vector in embedded_vectors.items():
                    vec = []
                    if "mean" in data:
                        vec.append(np.mean(vector, axis=0))
                    if "var" in data:
                        vec.append(np.var(vector, axis=0))
                    if "delta" in data:
                        vec.append(np.mean(librosa.feature.delta(vector),axis=0))
                    
                    vector = np.concatenate(vec, axis=0)

                    if file_name.split('/')[-1].split('.')[0] in input_name:
                        continue
                    
                    d = np.shape(vector)[0]
                    nb = len(embedded_vectors)
                
                euc_filename, _, cos_filename, _ = sim.mean_delta_similarity(current_embedding, embeddings, plot = "", data=data_type)
                euc_pred = np.append(euc_pred, int(euc_filename.split('/')[-2][:3]))
                
    # get the indices of the misclassified embeddings
    idxs = ((euc_pred == y_true) == False).nonzero()
                
    conf = confusion_matrix(y_true, euc_pred, labels = labels)
    
    acc = accuracy_score(y_true, euc_pred)
    
    if data_type == "normal":
        # save the results to .npy files
        np.save('output/comparisons/y_true.npy', y_true)
        np.save('output/comparisons/confusion_matrix_euclidean.npy', euc_conf)
        np.save('output/comparisons/confusion_matrix_cosine.npy', cos_conf)
        np.save('output/comparisons/indices_euclidean.npy', euc_idxs)
        np.save('output/comparisons/indices_cosine.npy', cos_idxs)
        np.save('output/comparisons/labels.npy', labels)
        np.save('output/comparisons/accuracy_euclidean.npy', euc_acc)
        np.save('output/comparisons/accuracy_cosine.npy', cos_acc)
    
    else:
        np.save('output/'+ data_name +'/comparisons/y_true.npy', y_true)
        np.save('output/'+ data_name +'/comparisons/confusion_matrix_euclidean.npy', euc_conf)
        np.save('output/'+ data_name +'/comparisons/confusion_matrix_cosine.npy', cos_conf)
        np.save('output/'+ data_name +'/comparisons/indices_euclidean.npy', euc_idxs)
        np.save('output/'+ data_name +'/comparisons/indices_cosine.npy', cos_idxs)
        np.save('output/'+ data_name +'/comparisons/labels.npy', labels)
        np.save('output/'+ data_name +'/comparisons/accuracy_euclidean.npy', euc_acc)
        np.save('output/'+ data_name +'/comparisons/accuracy_cosine.npy', cos_acc)
        np.save('output/'+ data_name +'/comparisons/euc_pred.npy', euc_pred)
        np.save('output/'+ data_name +'/comparisons/cos_pred.npy', cos_pred)
    
    return y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred
    
if __name__ == '__main__':
    data_names = ['mean', 'var', 'delta', 'meandelta', 'meanvar', 'vardelta', 'meanvardelta']
    data_types = [['mean'], ['var'], ['delta'], ['mean', 'delta'], ['mean', 'var'], ['var', 'delta'], ['mean', 'var', 'delta']]
        
    for i, data_type in enumerate(data_types):
        embedding_folders = 'output/embeddings/ECS50/'
        data_name = data_names[i]
        try:
            y_true = np.load('output/'+ data_name +'/comparisons/y_true.npy')
            euc_conf = np.load('output/'+ data_name +'/comparisons/confusion_matrix_euclidean.npy')
            euc_idxs = np.load('output/'+ data_name +'/comparisons/indices_euclidean.npy')
            cos_conf = np.load('output/'+ data_name +'/comparisons/confusion_matrix_cosine.npy')
            cos_idxs = np.load('output/'+ data_name +'/comparisons/indices_cosine.npy')
            labels = np.load('output/'+ data_name +'/comparisons/labels.npy')
            euc_acc = np.load('output/'+ data_name +'/comparisons/accuracy_euclidean.npy')
            cos_acc = np.load('output/'+ data_name +'/comparisons/accuracy_cosine.npy')
            euc_pred = np.load('output/'+ data_name +'/comparisons/euc_pred.npy')
            cos_pred = np.load('output/'+ data_name +'/comparisons/cos_pred.npy')
            print(f"\nLoaded numpy arrays from output folder {data_name}!\n")
            #raise Exception("Numpy arrays not found, creating them now...")
            
        except:
            print(f"\nNumpy arrays not found for data {data_name}, creating them now...\n")
            y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred = comparisons(embedding_folders, data_type, data_names[i])
        
        # present the accuracy of the euclidean and cosine similarity nicely in a table with the header of data type
        print(f"\n{data_names[i].upper()} DATA\n")
        # print(tabulate([['Euclidean', euc_acc], ['Cosine', cos_acc]], headers=['Similarity Type', 'Accuracy']))
        k = 5
        print(tabulate([['Euclidean Distance', euc_acc, ir.Precision(labels, euc_pred, y_true), ir.Recall(labels, euc_pred, y_true), 
                    ir.F1(labels, euc_pred, y_true), ir.MeanAveragePrecision(labels, euc_pred, y_true), ir.AccuracyAtK(labels, euc_pred, y_true, k)], 
                ['Cosine Distance', cos_acc, ir.Precision(labels, cos_pred, y_true), ir.Recall(labels, cos_pred, y_true), 
                    ir.F1(labels, cos_pred, y_true), ir.MeanAveragePrecision(labels, cos_pred, y_true), ir.AccuracyAtK(labels, cos_pred, y_true, k)]], 
                headers=['IR Type', 'Accuracy', 'Precision', 'Recall', 
                        'F1-Score', 'MAP', 'Accuracy @ K (K = 5)']))
        #plot_comparisons(y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, plot = ["all"], data_type = data_names[i])    
        