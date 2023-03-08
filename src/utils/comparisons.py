"""This file contains functions that compares all embeddings to each other
    and plots a matrix that shows how often the most similar embeddings beling
    to the same class/folder as the original embedding.
"""

import numpy as np
import os
from tqdm import tqdm
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score
import information_retrieval.metrics as ir

import similarity as sim

def savefig(fig, name, datatype="normal"):
    """Function that saves the current figure as a png file.
    
    Args:
        fig (figure): Figure that should be saved.
        name (str): Name of the plot that is saved. This name is used to create a folder.
    """
    # loop an arbitraty amount of times to save the plot to different names
    fig = fig
    for i in range(50):
        try:
            assert os.path.exists('output/comparisons/' + name)
            print('\n output/' + datatype + '/comparisons/' + name + ' already exists \n')
        except AssertionError:
            os.mkdir('output/' + datatype + '/comparisons/' + name)
        if not os.path.exists('output/' + datatype + '/comparisons/' + name + '/' + name + str(i) + datatype + '.png'):
            plt.savefig('output/' + datatype + '/comparisons/' + name + '/' + name + str(i) + datatype + '.png')
        else:
            print(f"\nFigure was already saved as {name + str(i) + datatype + '.png'}, continuing to next index")
            continue
        break 
        
def plot_comparisons(y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, plot="none", data_type="normal"):
    """Function that plots the confusion matrices and the misclassified embeddings.

    Args:
        y_true (array): Array containing the true labels of the embeddings.
        euc_conf (array): Array containing the confusion matrix of the euclidean distance.
        cos_conf (array): Array containing the confusion matrix of the cosine similarity.
        euc_idxs (array): Array containing the indices of the misclassified embeddings
        cos_idxs (array): Array containing the indices of the misclassified embeddings
        labels (array): Array containing the labels of the embeddings.
        plot (array, optional): Array of strings that defines which plots should be plotted. Defaults to "none".
    """
    #assert plot in ["none", "all", "confusion_matrix", "misclassified", "misclassified_per_class", "misclassified_audioset_missing"], "Invalid plot argument. Must be 'none', 'all', 'confusion_matrix', 'misclassified', 'misclassified_per_class' or 'misclassified_audioset_missing'"
    if plot == ["all"]:
        #plot = ["confusion_matrix", "misclassified", "misclassified_per_class", "misclassified_audioset_missing", "most_missclassified"]
        plot = ["confusion_matrix", "misclassified", "misclassified_per_class", "most_missclassified"]
    
    if "confusion_matrix" in plot:
        # plot the matrices as subplots
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Euclidean distance')
        plt.imshow(euc_conf)
        plt.subplot(1, 2, 2)
        plt.title('Cosine similarity')
        plt.imshow(cos_conf)
        
        name = "confusion_matrix"
        savefig(plt, name, data_type)
    
    if "misclassified" in plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Euclidean distance')
        plt.bar(np.unique(y_true[euc_idxs], return_counts=True)[0], np.unique(y_true[euc_idxs], return_counts=True)[1])
        plt.subplot(1, 2, 2)
        plt.title('Cosine similarity')
        plt.bar(np.unique(y_true[cos_idxs], return_counts=True)[0], np.unique(y_true[cos_idxs], return_counts=True)[1])
        
        name = "misclassified"
        savefig(plt, name, data_type)
    
    if "misclassified_per_class" in plot:
        # create a 1x5 plot that shows the amount of misclassified embeddings for each class
        plt.figure(figsize=(16, 8))
        for i in range(50):
            plt.subplot(5, 10, i+1)
            plt.tight_layout()
            plt.title('Class ' + str(labels[i]))
            try:
                plt.bar(['Euc', 'Cos'], [np.unique(y_true[euc_idxs], return_counts=True)[1][i], np.unique(y_true[cos_idxs], return_counts=True)[1][i]])
            except:
                # wierd error that occurs when the class is not in the misclassified embeddings
                # TODO: Fix this error
                
                # for label in labels:
                #     if label not in np.unique(y_true[euc_idxs], return_counts=True)[0]:
                #         euc_temp = 0
                #     else:
                #         euc_temp = np.unique(y_true[euc_idxs], return_counts=True)[1][i]
                #     if label not in np.unique(y_true[cos_idxs], return_counts=True)[0]:
                #         cos_temp = 0
                #     else:
                #         cos_temp = np.unique(y_true[cos_idxs], return_counts=True)[1][i]
                
                # Error handling for now
                euc_temp, cos_temp = 0, 0
                plt.bar(['Euc', 'Cos'], [euc_temp, cos_temp])
        
        name = "misclassified_per_class"
        savefig(plt, name, data_type)
            
    if "misclassified_audioset_missing" in plot:
        audioset_labels = [104, 107, 310, 405, 406]
        # exclude the audioset labels from the labels
        labels2 = np.delete(labels, np.where(np.isin(labels, audioset_labels)))
        # select 5 other random labels from labels
        random_labels = np.sort(np.random.choice(labels2, 5, replace=False))
        # create a 1x5 plot that shows the amount of misclassified embeddings for each class
        plt.figure(figsize=(16, 8))
        plt.tight_layout()
        plt.title("Misclassified embeddings per class")
        for i in range(len(audioset_labels)):
            plt.subplot(2, 5, i+1)
            #plt.suptitle("Classes absent from AudioSet")
            plt.title('Class ' + str(audioset_labels[i]))
            # get the indices of the audioset labels
            euc_mask = np.unique(y_true[euc_idxs], return_counts=True)[0] == audioset_labels[i]
            cos_mask = np.unique(y_true[cos_idxs], return_counts=True)[0] == audioset_labels[i]
            plt.bar(['Euc', 'Cos'], [int(np.unique(y_true[euc_idxs], return_counts=True)[1][euc_mask]), int(np.unique(y_true[cos_idxs], return_counts=True)[1][cos_mask])], color=['indianred', 'cornflowerblue'])
            
            plt.subplot(2, 5, i+5+1)
            #plt.suptitle("Classes present in AudioSet")
            plt.title('Class ' + str(random_labels[i]))
            # get the indices of the audioset labels
            euc_rand_mask = np.unique(y_true[euc_idxs], return_counts=True)[0] == random_labels[i]
            cos_rand_mask = np.unique(y_true[cos_idxs], return_counts=True)[0] == random_labels[i]
            plt.bar(['Euc', 'Cos'], [int(np.unique(y_true[euc_idxs], return_counts=True)[1][euc_rand_mask]), int(np.unique(y_true[cos_idxs], return_counts=True)[1][cos_rand_mask])], color=['indianred', 'cornflowerblue'])
        
        name = "misclassified_audioset_missing"
        savefig(plt, name, data_type) 
        
    if "most_misclassified" in plot:
        classes, counts = np.unique(y_true[euc_idxs], return_counts=True)

        # Set up the figure with two axes
        f, (ax, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(10, 8), sharex=False)

        # Plot the data on both axes
        ax.bar(classes, counts, color='lightblue')
        ax2.bar(classes, counts, color='lightblue')
        ax3.bar(classes, counts, color='lightblue')
        ax4.bar(classes, counts, color='lightblue')
        ax5.bar(classes, counts, color='lightblue')

        # plot the mean of the misclassified classes in each class range
        mean1 = np.mean(counts[0:10])
        mean2 = np.mean(counts[10:20])
        mean3 = np.mean(counts[20:30])
        mean4 = np.mean(counts[30:40])
        mean5 = np.mean(counts[40:50])

        # plot them as a red horizontal line
        ax.axhline(y=mean1, color='r', linestyle='-')
        ax2.axhline(y=mean2, color='r', linestyle='-')
        ax3.axhline(y=mean3, color='r', linestyle='-')
        ax4.axhline(y=mean4, color='r', linestyle='-')
        ax5.axhline(y=mean5, color='r', linestyle='-')

        # label the red horizontal lines with the mean
        # the label should be black text with the greek letter mu and the mean rounded to 2 decimals
        ax.text(109, mean1+0.5, r'$\mu$ =' + str(round(mean1, 2)), horizontalalignment='center', verticalalignment='center', color='black')
        ax2.text(209, mean2+0.5, r'$\mu$ =' + str(round(mean2, 2)), horizontalalignment='center', verticalalignment='center', color='black')
        ax3.text(309, mean3+0.5, r'$\mu$ =' + str(round(mean3, 2)), horizontalalignment='center', verticalalignment='center', color='black')
        ax4.text(409, mean4+0.5, r'$\mu$ =' + str(round(mean4, 2)), horizontalalignment='center', verticalalignment='center', color='black')
        ax5.text(509, mean5+0.5, r'$\mu$ =' + str(round(mean5, 2)), horizontalalignment='center', verticalalignment='center', color='black')

        # Zoom-in / limit the view to different portions of the data
        ax.set_xlim(100, 111)  
        ax2.set_xlim(200, 211)
        ax3.set_xlim(300, 311) 
        ax4.set_xlim(400, 411)
        ax5.set_xlim(500, 511)

        ax.set_xticks(range(101, 111, 2))
        ax2.set_xticks(range(201, 211, 2))
        ax3.set_xticks(range(301, 311, 2))
        ax4.set_xticks(range(401, 411, 2))
        ax5.set_xticks(range(501, 511, 2))

        # add titles to each subplot with their class range
        ax.set_title('Classes: 101-110')
        ax2.set_title('Classes: 201-210')
        ax3.set_title('Classes: 301-310')
        ax4.set_title('Classes: 401-410')
        ax5.set_title('Classes: 501-510')

        # Set the x-axis label and the plot title
        f.suptitle('Number of times the input vector from class X is wrongly classified.', fontsize=14)

        # Adjust the spacing between the plots
        f.subplots_adjust(hspace=.05)
        
        name = "misclassified_classes"
        savefig(f, name, data_type)

def comparisons(data_path='output/embeddings/ECS50/', data_type = "normal", data_name = "mean"):
    """Function that compares all embeddings to each other and plots a matrix that shows how often
    the most similar embeddings beling to the same class/folder as the original embedding.

    Args:
        data_path (str, optional): Path of the embeddings. Defaults to 'output/embeddings/'.
    """
    y_true, euc_pred, cos_pred = np.array([]), np.array([]), np.array([])
    labels = [int(folder[:3]) for folder in os.listdir(data_path) if folder != '.DS_Store']
    
    embeddings = []
    # get all the embeddings
    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
        embeddings.append(['output/embeddings/ECS50/' + folder + '/' + file for file in os.listdir('output/embeddings/ECS50/' + folder)])
        
    # flatten the list
    embeddings = [item for sublist in embeddings for item in sublist]
    
    for i, folder in tqdm(enumerate(os.listdir(data_path)), total = len(os.listdir(data_path))):
        if folder == '.DS_Store':
            continue
        for file in os.listdir(data_path + folder):
            if file.endswith('.npy') and not file.endswith('_trimmed.npy'):
                # load the current embedding
                current_embedding = str(data_path + folder + '/' + file)
                y_true = np.append(y_true, int(folder[:3]))
                # compare the current embedding to all other embeddings
                if data_type == "normal" or data_type == "trimmed":
                    euc_filename, _, cos_filename, _ = sim.ECS_similarity(current_embedding, embeddings, plot = "")
                else:
                    euc_filename, _, cos_filename, _ = sim.mean_delta_similarity(current_embedding, embeddings, plot = "", data=data_type)
                euc_pred = np.append(euc_pred, int(euc_filename.split('/')[-2][:3]))
                cos_pred = np.append(cos_pred, int(cos_filename.split('/')[-2][:3]))
                
    # get the indices of the misclassified embeddings
    euc_idxs = ((euc_pred == y_true) == False).nonzero()
    cos_idxs = ((cos_pred == y_true) == False).nonzero()
                
    euc_conf = confusion_matrix(y_true, euc_pred, labels = labels)
    cos_conf = confusion_matrix(y_true, cos_pred, labels = labels)
    
    euc_acc = accuracy_score(y_true, euc_pred)
    cos_acc = accuracy_score(y_true, cos_pred)    
    
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
        
        if data_type == 'normal':
            embedding_folders = 'output/embeddings/ECS50/'
            
            # check if the numpy arrays exist, if not, create them
            try:
                # y_true = np.load('output/comparisons/y_true.npy')
                # euc_conf = np.load('output/comparisons/confusion_matrix_euclidean.npy')
                # euc_idxs = np.load('output/comparisons/indices_euclidean.npy')
                # cos_conf = np.load('output/comparisons/confusion_matrix_cosine.npy')
                # cos_idxs = np.load('output/comparisons/indices_cosine.npy')
                # labels = np.load('output/comparisons/labels.npy')
                # euc_acc = np.load('output/comparisons/accuracy_euclidean.npy')
                # cos_acc = np.load('output/comparisons/accuracy_cosine.npy')
                # print(f"\nLoaded numpy arrays from output folder!\n")
                raise Exception("Creating Numpy Arrays")
                
            except:
                print(f"\nNumpy arrays not found, creating them now...\n")
                y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc = comparisons(embedding_folders, data_type)
            
            # present the accuracy of the euclidean and cosine similarity nicely in a table
            print(f"\n{data_type.upper()} DATA\n")
            print(tabulate([['Euclidean', euc_acc], ['Cosine', cos_acc]], headers=['Similarity Type', 'Accuracy']))
            
            plot_comparisons(y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, plot = ["all"], data_type = data_type)
            # bad practice, remove when debugging is done
        
        elif data_type == 'trimmed':
            embedding_folders = 'output/embeddings/ECS50_trimmed/'
            
            try:
                y_true = np.load('output/trimmed/comparisons/y_true.npy')
                euc_conf = np.load('output/trimmed/comparisons/confusion_matrix_euclidean.npy')
                euc_idxs = np.load('output/trimmed/comparisons/indices_euclidean.npy')
                cos_conf = np.load('output/trimmed/comparisons/confusion_matrix_cosine.npy')
                cos_idxs = np.load('output/trimmed/comparisons/indices_cosine.npy')
                labels = np.load('output/trimmed/comparisons/labels.npy')
                euc_acc = np.load('output/trimmed/comparisons/accuracy_euclidean.npy')
                cos_acc = np.load('output/trimmed/comparisons/accuracy_cosine.npy')
                print(f"\nLoaded numpy arrays from output folder!\n")
                #raise Exception("Numpy arrays not found, creating them now...")
                
            except:
                print(f"\nNumpy arrays from trimmed data not found, creating them now...\n")
                y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc = comparisons(embedding_folders, data_type = data_type)
                
            # present the accuracy of the euclidean and cosine similarity nicely in a table
            print(f"\n{data_type.upper()} DATA\n")
            print(tabulate([['Euclidean', euc_acc], ['Cosine', cos_acc]], headers=['Similarity Type', 'Accuracy']))
            
            plot_comparisons(y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, plot = ["all"], data_type = data_type)
            
        elif data_type == 'mean_delta':
            embedding_folders = 'output/embeddings/ECS50/'
            y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred = comparisons(embedding_folders, data_type = data_type)
            
            # present the accuracy of the euclidean and cosine similarity nicely in a table with the header of data type
            print(f"\n{data_names[i].upper()} DATA\n")
            print(tabulate([['Euclidean', euc_acc], ['Cosine', cos_acc]], headers=['Similarity Type', 'Accuracy']))
            
            plot_comparisons(y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, plot = ["all"], data_type = data_type)
        
        else:
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
        