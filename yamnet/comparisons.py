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
from scipy.stats import ttest_ind

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
            assert os.path.exists('output/' + datatype + '/comparisons/' + name)
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
        # select 5 other random labels from labelsl
        # exclude label 210 from the random labels
        labels2 = np.delete(labels2, np.where(labels2 == 210))
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

def comparisons(data_path='output/embeddings/ECS50/', data_type = "mean", data_name = "mean"):
    """Function that compares all embeddings to each other and plots a matrix that shows how often
    the most similar embeddings beling to the same class/folder as the original embedding.

    Args:
        data_path (str, optional): Path of the embeddings. Defaults to 'output/embeddings/'.
        data_type (str, optional): Type of the embeddings. Defaults to "mean".
        data_name (str, optional): Name of the embeddings. Defaults to "mean".
        
    Returns:
        predictions and metrics generated in the comparisons
    """
    # initialize variables
    top_k = 5
    y_true, euc_pred, cos_pred, euc_dists, cos_dists = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    euc_top_k_preds, cos_top_k_preds = [], []
    labels = [int(folder[:3]) for folder in os.listdir(data_path) if folder != '.DS_Store']
    
    embeddings = []
    # get all the embeddings
    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
        embeddings.append(['output/embeddings/ECS50/' + folder + '/' + file for file in os.listdir('output/embeddings/ECS50/' + folder)])
        
    # flatten the list
    embeddings = [item for sublist in embeddings for item in sublist]
    
    # loop through all the embeddings
    for i, folder in tqdm(enumerate(os.listdir(data_path)), total = len(os.listdir(data_path))):
        # skip the .DS_Store file (mac thing)
        if folder == '.DS_Store':
            continue
        # loop through all the embeddings in the current folder
        for file in os.listdir(data_path + folder):
            if file.endswith('.npy') and not file.endswith('_trimmed.npy'):
                # load the current embedding
                current_embedding = str(data_path + folder + '/' + file)
                y_true = np.append(y_true, int(folder[:3]))
                # compare the current embedding to all other embeddings
                if data_type == "normal" or data_type == "trimmed":
                    euc_filename, euc_distances, cos_filename, cos_distances = sim.ECS_similarity(current_embedding, embeddings, plot = "")
                else:
                    euc_filename, euc_distances, cos_filename, cos_distances = sim.mean_delta_similarity(current_embedding, embeddings, plot = "", data=data_type, VAE=True)
                euc_pred = np.append(euc_pred, int(euc_filename.split('/')[-2][:3]))
                cos_pred = np.append(cos_pred, int(cos_filename.split('/')[-2][:3]))
                
                euc_top_k_preds.append([int([key for key, val in euc_distances.items() if val == searchval][0].split('/')[-2][:3]) for searchval in np.sort(np.array(list(euc_distances.values())))[:top_k]])
                cos_top_k_preds.append([int([key for key, val in cos_distances.items() if val == searchval][0].split('/')[-2][:3]) for searchval in np.sort(np.array(list(cos_distances.values())))[:top_k]])
                
    # get the indices of the misclassified embeddings
    euc_idxs = ((euc_pred == y_true) == False).nonzero()
    cos_idxs = ((cos_pred == y_true) == False).nonzero()
                
    euc_conf = confusion_matrix(y_true, euc_pred)
    cos_conf = confusion_matrix(y_true, cos_pred)
    
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
        ars =     [y_true,   euc_conf,   cos_conf,   euc_idxs,   cos_idxs,   labels,   euc_acc,   cos_acc,   euc_pred,   cos_pred,   euc_top_k_preds,   cos_top_k_preds]
        arnames = ["y_true", "euc_conf", "cos_conf", "euc_idxs", "cos_idxs", "labels", "euc_acc", "cos_acc", "euc_pred", "cos_pred", "euc_top_k_preds", "cos_top_k_preds"]
        
        for i, ar in enumerate(ars):
            np.save('output/'+ data_name +'/comparisons/' + arnames[i] + '.npy', ar)
    
    return y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds

def t_test_missing_labels(y_true, euc_idxs, cos_idxs):
    """ Function that calculates a t-test stastistic to assess whether some classes are misclassified more often than others. """
    
    # get the missing labels
    missing_labels = [104, 107, 310, 405, 406]
    
    # get the accuracies of the missing labels
    missing_labels_acc_euc, missing_labels_acc_cos = [], []
    for label in missing_labels:
        missing_labels_acc_euc.append(np.sum(y_true[euc_idxs] == label) / np.sum(y_true == label))
        missing_labels_acc_cos.append(np.sum(y_true[cos_idxs] == label) / np.sum(y_true == label))
        
    other_labels = [i for i in np.unique(y_true) if i not in missing_labels]
    
    equal_var = True
    
    print(f"Equal variance assumption: {equal_var}\n")
    
    other_labels_acc_euc, other_labels_acc_cos = [], []
    for label in other_labels:
        other_labels_acc_euc.append(np.sum(y_true[euc_idxs] == label) / np.sum(y_true == label))
        other_labels_acc_cos.append(np.sum(y_true[cos_idxs] == label) / np.sum(y_true == label))
        
    # tabluate mean and std of the accuracies
    print("Mean and standard deviation of the classification accuracies of the missing labels:")
    print(tabulate([["Euclidian Distance (Missing Labels)", np.mean(missing_labels_acc_euc).round(3), np.std(missing_labels_acc_euc).round(3), np.mean(other_labels_acc_euc).round(3), np.std(other_labels_acc_euc).round(3)],
                    ["Cosine Similarity (Missing Labels)", np.mean(missing_labels_acc_cos).round(3), np.std(missing_labels_acc_cos).round(3), np.mean(other_labels_acc_cos).round(3), np.std(other_labels_acc_cos).round(3)]],
                   headers=["Metric", "Mean (Missing)", "Mean (Other)", "Std. (Missing)", "Std. (Other)"]))
    print("\n")
                   
    
    # perform t-test
    t_stat_euc, p_val_euc = ttest_ind(missing_labels_acc_euc, other_labels_acc_euc, equal_var=equal_var)
    t_stat_cos, p_val_cos = ttest_ind(missing_labels_acc_cos, other_labels_acc_cos, equal_var=equal_var)
        
    # Print results
    print(f"T-test results (Euclidian Distance): t = {t_stat_euc.round(3)}, p = {p_val_euc.round(3)}")
    if p_val_euc < 0.05:
        print("The difference in classification accuracy is statistically significant.")
    else:
        print("The difference in classification accuracy is not statistically significant.")
        
    print(f"\n\nT-test results (Cosine Similarity): t = {t_stat_cos.round(3)}, p = {p_val_cos.round(3)}")
    if p_val_cos < 0.05:
        print("The difference in classification accuracy is statistically significant.")
    else:
        print("The difference in classification accuracy is not statistically significant.")
        
    print("\n")
        
    return p_val_euc, p_val_cos

def tabulate_results(euc_acc, cos_acc, labels, y_true, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds, k, savefolder, mode="superclass"):
    print(f"Mode: {mode}\n")
    if mode == "superclass":
        labels = [int(str(label)[:1]) for label in labels]
        y_true = [int(str(label)[:1]) for label in y_true]
        euc_pred = [int(str(label)[:1]) for label in euc_pred]
        cos_pred = [int(str(label)[:1]) for label in cos_pred]
        euc_acc = ir.Accuracy(labels = labels, y_true = y_true, pred = euc_pred)
        cos_acc = ir.Accuracy(labels = labels, y_true = y_true, pred = cos_pred)
        euc_top_k_preds = [[int(str(label)[:1]) for label in top_k_preds] for top_k_preds in euc_top_k_preds]
        cos_top_k_preds = [[int(str(label)[:1]) for label in top_k_preds] for top_k_preds in cos_top_k_preds]
    elif mode == "subclass":
        euc_acc = ir.Accuracy(labels = labels, y_true = y_true, pred = euc_pred)
    elif mode == "baseline_superclass":
        # make random predictions of 1-5
        euc_pred = [np.random.randint(1,5) for i in range(len(euc_pred))]
        cos_pred = [np.random.randint(1,5) for i in range(len(cos_pred))]
        cos_top_k_preds = [[np.random.randint(1,5) for i in range(k)] for j in range(len(cos_top_k_preds))]
        euc_top_k_preds = [[np.random.randint(1,5) for i in range(k)] for j in range(len(euc_top_k_preds))]
    map_string = 'MAP @ ' + str(k)
    acc_string = 'Acc @ ' + str(k)
    tab = tabulate([['Euc. Dist.', euc_acc, ir.Precision(labels = labels, y_true = y_true, pred = euc_pred), 
                    ir.Recall(labels = labels, y_true = y_true, pred = euc_pred), 
                    ir.F1(labels = labels, y_true = y_true, pred = euc_pred), 
                    ir.MeanAveragePrecision(labels = labels, y_true = y_true, top_k_preds = euc_top_k_preds, k=k), 
                    ir.AccuracyAtK(labels = labels, y_true = y_true, top_k_preds = euc_top_k_preds, k=k)], 
                    
                    ['Cos. Sim.', cos_acc, ir.Precision(labels = labels, y_true = y_true, pred = cos_pred),
                    ir.Recall(labels = labels, y_true = y_true, pred = cos_pred), 
                    ir.F1(labels = labels, y_true = y_true, pred = cos_pred), 
                    ir.MeanAveragePrecision(labels = labels, y_true = y_true, top_k_preds = cos_top_k_preds, k=k), 
                    ir.AccuracyAtK(labels = labels, y_true = y_true, top_k_preds = cos_top_k_preds, k=k)]
                    ], 
        headers=[' ', 'Accuracy', 'Precision', 'Recall', 
                'F1-Score', map_string, acc_string], tablefmt='latex_raw')
    
    tab2 = tabulate([['Euc. Dist.', 
                    ir.MeanAveragePrecision(labels = labels, y_true = y_true, top_k_preds = euc_top_k_preds, k=4),
                    ir.AccuracyAtK(labels = labels, y_true = y_true, top_k_preds = euc_top_k_preds, k=4),
                    ir.MeanAveragePrecision(labels = labels, y_true = y_true, top_k_preds = euc_top_k_preds, k=3),
                    ir.AccuracyAtK(labels = labels, y_true = y_true, top_k_preds = euc_top_k_preds, k=3),
                    ir.MeanAveragePrecision(labels = labels, y_true = y_true, top_k_preds = euc_top_k_preds, k=2),
                    ir.AccuracyAtK(labels = labels, y_true = y_true, top_k_preds = euc_top_k_preds, k=2)],
                    ['Cos. Sim.',
                    ir.MeanAveragePrecision(labels = labels, y_true = y_true, top_k_preds = cos_top_k_preds, k=4),
                    ir.AccuracyAtK(labels = labels, y_true = y_true, top_k_preds = cos_top_k_preds, k=4),
                    ir.MeanAveragePrecision(labels = labels, y_true = y_true, top_k_preds = cos_top_k_preds, k=3),
                    ir.AccuracyAtK(labels = labels, y_true = y_true, top_k_preds = cos_top_k_preds, k=3),
                    ir.MeanAveragePrecision(labels = labels, y_true = y_true, top_k_preds = cos_top_k_preds, k=2),
                    ir.AccuracyAtK(labels = labels, y_true = y_true, top_k_preds = cos_top_k_preds, k=2)]],
                    headers = [' ', 'MAP @ 4', 'Acc @ 4', 'MAP @ 3', 'Acc @ 3', 'MAP @ 2', 'Acc @ 2'], tablefmt='latex_raw')
    
    with open(savefolder + '/results_' + str(mode)+ '.txt', 'w') as f:
        f.write(tab)
        f.write(tab2)

    return tab, tab2
    
if __name__ == '__main__':
    # set numpy seed
    np.random.seed(42)
    
    # Set up comparisons for different data types
    data_names = ['mean', 'var', 'delta', 'meandelta', 'meanvar', 'vardelta', 'meanvardelta']
    data_types = [['mean'], ['var'], ['delta'], ['mean', 'delta'], ['mean', 'var'], ['var', 'delta'], ['mean', 'var', 'delta']]
    for i, data_type in enumerate(data_types):
        print(f"Data type: {data_type}")
        
        # path of saved embeddings, generated by embeddings.py
        embedding_folders = 'output/embeddings/ECS50/'
        data_name = data_names[i]
        
        try:
            # load saved embeddings if they exist
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
            euc_top_k_preds = np.load('output/'+ data_name +'/comparisons/euc_top_k_preds.npy')
            cos_top_k_preds = np.load('output/'+ data_name +'/comparisons/cos_top_k_preds.npy')
            print(f"\nLoaded numpy arrays from output folder {data_name}!\n")
            
        except Exception as e:
            # create them if they don't exist
            print(f"\n{e}\n")
            print(f"\nNumpy arrays not found for data {data_name}, creating them now...\n")
            y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds = comparisons(embedding_folders, data_type, data_names[i])
        
        # present the accuracy of the euclidean and cosine similarity nicely in a table with the header of data type
        print(f"\n{data_names[i].upper()} DATA\n")
        # Initialize k for top-k accuracy
        k = 5
        savefolder = f"output/{data_name}/comparisons"
        print(tabulate_results(euc_acc, cos_acc, labels, y_true, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds, k, savefolder, mode="superclass"))
        print(tabulate_results(euc_acc, cos_acc, labels, y_true, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds, k, savefolder, mode="subclass"))
        
        plot_comparisons(y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, plot = ["misclassified_audioset_missing"], data_type = data_names[i])    
        
        #t_test_missing_labels(y_true, euc_idxs, cos_idxs)
    