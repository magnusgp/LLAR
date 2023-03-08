"""This file performs PCA on the embeddings and plots the results.
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

from sklearn.feature_selection import VarianceThreshold

def embeddingPCA(embeddings, dim):
    """Performs PCA on the embeddings and plots the results.

    Args:
        embeddings (list): List of all the embeddings.
    """
    # load the embeddings as numpy arrays
    load_embeds, labels = [], []
    for embedding in embeddings:
        embed = np.load(embedding, allow_pickle=True)
        if np.shape(embed) != (10, 1024):
            continue
        labels.append(embedding.split('/')[-2][:3])
        load_embeds.append(embed)
    # perform PCA
    pca = PCA(n_components=10)
    nsamples, nx, ny = np.shape(load_embeds)
    load_embeds_PCA = np.reshape(load_embeds, (nsamples, nx*ny))
    pca.fit(load_embeds_PCA)
    # get the new embeddings
    load_embeds_PCA = pca.transform(load_embeds_PCA)
    # make sure that all labels have the same color for plotting
    # make an array with 50 colors for the 50 classes
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'pink', 'brown', 'grey', 'olive', 'cyan', 'tan', 'lime', 'teal', 'lavender', 'turquoise', 'darkgreen', 'gold', 'darkred', 'darkblue', 'darkorange', 'darkgrey', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkyellow', 'darkkhaki', 'darkolivegreen', 'darkorchid', 'darkseagreen', 'darkslateblue', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'greenyellow']
    color = []
    unique_labels = np.unique(labels)
    for label in labels:
        label_idx = np.where(unique_labels == label)
        color.append(colors[label_idx[0][0]])
        
    # turn labels into integers
    labels = [int(label) for label in labels]
    plt.figure(figsize=(8,5))
    if dim == 2:
        # plot the embeddings 2D with label colors for the 50 classes
        plt.subplot(1, 2, 1)
        plt.title('PCA of the embeddings')
        plt.scatter(load_embeds_PCA[:, 0], load_embeds_PCA[:, 1], c=labels)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

    elif dim == 3:
        # plot the embeddings 3D with label colors for the 50 classes
        #plt.subplot(1, 2, 1)
        plt.title('PCA of the embeddings')
        ax = plt.axes(projection='3d')
        ax.scatter3D(load_embeds_PCA[:, 0], load_embeds_PCA[:, 1], load_embeds_PCA[:, 2], c=labels)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    # plot the cumulative explained variance
    # plt.subplot(1, 2, 2)
    # plt.title('Cumulative explained variance')
    # plt.plot(exp_var_cumul)
    # plt.xticks(np.arange(0, 10, 1))
    # plt.show()
    
    
def embeddingPCA_manual(embeddings, dim):
    """Performs PCA on the embeddings without sklearn.
       Then plots the results.

    Args:
        embeddings (list): List of all the embeddings.
        dim (int): Dimension of the plot.
    """
    load_embeds, labels = [], []
    max_embed_len = max(np.unique([len(np.load(embedding)) for embedding in embeddings], return_counts=True)[0])
    for embedding in embeddings:
        embed = np.load(embedding, allow_pickle=True)
        if np.shape(embed) != (10,1024):
            # zero pad the embeddings so that the first dimension is the same for all embeddings
            continue
        labels.append(embedding.split('/')[-2][:3])
        embed = np.reshape(embed, (10*1024))
        load_embeds.append(embed)
        
    # plot the variance of the embeddings
    plt.figure(figsize=(8,5))
    plt.title('Variance of the embeddings')
    plt.hist(np.var(load_embeds, axis=1))
    plt.show()
    
    # initialize figure for plots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    
    figvar, axsvar = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    
    # select only the embeddings with a variance higher than 1.0
    thresholds = [0.00, 0.70, 0.80, 0.90, 1.00, 1.10]
    for i, threshold in enumerate(thresholds):
        selector = VarianceThreshold(threshold=threshold)
        load_embeds = selector.fit_transform(load_embeds)
        
        # perform PCA
        
        # center the data
        embed_meaned = load_embeds - np.mean(load_embeds, axis=0)
        
        std = np.std(embed_meaned, axis=0)
        
        # make sure that the standard deviation is not 0 by adding a small value where it is 0
        std[std == 0] = 1e-10
        
        # standarisize the data
        embed_standardized = embed_meaned / std
        
        # calculate the covariance matrix for the standarized embeddings
        cov_mat = np.cov(embed_standardized, rowvar=False)
        
        # calculate the eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov_mat)
        
        # sort the eigenvalues and eigenvectors in descending order
        sort_idx = np.argsort(eigenvals)[::-1]
        
        sort_eigenvals = eigenvals[sort_idx]
        sort_eigenvecs = eigenvecs[:, sort_idx]
        
        n_components = dim
        
        # transform the data
        embed_transformed = np.dot(sort_eigenvecs[:, :n_components].T, embed_standardized.T).T
        
        #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'pink', 'brown', 'grey', 'olive', 'cyan', 'tan', 'lime', 'teal', 'lavender', 'turquoise', 'darkgreen', 'gold', 'darkred', 'darkblue', 'darkorange', 'darkgrey', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkyellow', 'darkkhaki', 'darkolivegreen', 'darkorchid', 'darkseagreen', 'darkslateblue', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'greenyellow']
        colors = ['b', 'g', 'r', 'c', 'm']
        color = []
        parent_labels = np.unique([int(np.unique(label)) // 100 for label in labels])
        unique_labels = np.unique(labels)
        for label in labels:
            # make the labels belong to their parent class, where labels 101-110 belong to class 1, 201-210 to class 2, etc.
            # label = int(unique_labels)
            parent_label = int(label) // 100
            parent_label_idx = np.where(parent_labels == parent_label)
            color.append(colors[parent_label_idx[0][0]])
            
        # turn labels into integers
        labels = [int(label) for label in labels]
        #f, axarr = plt.subplots(2, 3)
        # plt.figure(figsize=(10,8))
        # plt.tight_layout()
        # plt.title("PCA of the embeddings with variance thresholds")
        dim = 3
        if dim == 2:
            ax = axs[i//3, i%3]
            ax.scatter(embed_transformed[:, 0], embed_transformed[:, 1], color=color)
            ax.set_title(f"Variance threshold={threshold}")
            # plot the embeddings 2D with label colors for the 50 classes
            #plt.subplot(2, 3, i+1)
            #plt.suptitle('Threshold = ' + str(thresholds[i]))
            #plt.scatter(embed_transformed[:, 0], embed_transformed[:, 1], color=color)
            #plt.xlabel('PC1')
            #plt.ylabel('PC2')
            # if i in [0, 1, 2]:
            #     axarr[0, i].set_title('Threshold = ' + str(thresholds[i]))
            #     axarr[0, i].scatter(embed_transformed[:, 0], embed_transformed[:, 1], color=color)
            # elif i in [3, 4, 5]:
            #     axarr[1, i-3].set_title('Threshold = ' + str(thresholds[i]))
            #     axarr[1, i-3].scatter(embed_transformed[:, 0], embed_transformed[:, 1], color=color)

        elif dim == 3:
            # plot the embeddings 3D with label colors for the 50 classes
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
            ax.scatter(embed_transformed[:, 0], embed_transformed[:, 1], embed_transformed[:, 2], color=color)
            ax.set_title(f"Variance threshold={threshold}")
            #plt.subplot(1, 2, 1)
            # plt.title('PCA of the embeddings')
            # ax = plt.axes(projection='3d')
            # ax.scatter3D(embed_transformed[:, 0], embed_transformed[:, 1], embed_transformed[:, 2], color=color)
            # plt.xlabel('PC1')
            # plt.ylabel('PC2')
            
    plt.tight_layout()
    plt.show()
    
    
    
    
    
if __name__ == "__main__":
    
    data_path = 'output/embeddings/ECS50/'
    embeddings = []
    # get all the embeddings
    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
        embeddings.append([data_path + folder + '/' + file for file in os.listdir(data_path + folder) if file.endswith('_trimmed.npy')])
    
    embeddings = [item for sublist in embeddings for item in sublist]
    
    dim = 10
    
    #embeddingPCA(embeddings, dim)
    embeddingPCA_manual(embeddings, dim)