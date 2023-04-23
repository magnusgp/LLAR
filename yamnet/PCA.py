"""This file performs PCA on the embeddings and plots the results.
"""
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def embeddingPCA(embeddings, dim):
    """Performs PCA on the embeddings and plots the results.

    Args:
        embeddings (list): List of all the embeddings.
        dim (int): Dimension of the PCA, either 2 or 3.
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
    #plot the cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.title('Cumulative explained variance')
    plt.plot(exp_var_cumul)
    plt.xticks(np.arange(0, 10, 1))
    plt.show()
    
    
def whiten(X,fudge=1E-18):
    #Attempt to perform PCA whitening on the embeddings.
    #Did not work properly and is therefore not used.
   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V, D), V.T)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white, W
    
def embeddingPCA_manual(embeddings, dim):
    """Performs PCA on the embeddings without sklearn.
       Then plots the results and fits linear classifiers on the embeddings.

    Args:
        embeddings (list): List of all the embeddings.
        dim (int): Dimension of the plot.
    """
    load_embeds, labels = [], []
    max_embed_len = max(np.unique([len(np.load(embedding)) for embedding in embeddings], return_counts=True)[0])
    for embedding in embeddings:
        embed = np.load(embedding, allow_pickle=True)
        labels.append(embedding.split('/')[-2][:3])
        embed = np.reshape(embed, (1024))
        load_embeds.append(embed)
        
    # plot the variance of the embeddings
    plt.figure(figsize=(8,5))
    plt.title('Variance of the embeddings')
    plt.hist(np.var(load_embeds, axis=1), color='lightblue', edgecolor='black', bins=20)
    # plot a red vertical line at the mean of the variance
    plt.axvline(np.mean(np.var(load_embeds, axis=1)), color='r', linestyle='dashed', linewidth=2)
    # print the mean of the variance as text besides the red vertical line
    plt.text(np.mean(np.var(load_embeds, axis=1)) + 0.0005, 750, r'$\mu$ = {:.5f}'.format(np.mean(np.var(load_embeds, axis=1))))
    # give the plot a title and labels
    plt.xlabel('Variance')
    plt.ylabel('Count')
    plt.title('Variance of the embeddings')
    plt.savefig('variance_embeds.png')
    
    # initialize figure for plots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    
    figvar, axsvar = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    
    scores = pd.DataFrame(columns=['threshold', 'num_features', 'clf', 'clfCV', 'reg', 'bayes', 'svc'])
    
    # threshold ranges from 0.0000 to 0.0025 with a step size of 0.05
    thresholds = np.arange(0.0000, 0.0025, 0.0001)
    
    # copy load_embeds
    load_embeds_cpy = load_embeds.copy()
    
    #thresholds = [0.00]
    for i, threshold in enumerate(thresholds):
        print(f'Performing PCA with threshold {threshold}\n')
        selector = VarianceThreshold(threshold=threshold)
        # reinitialize load_embeds
        load_embeds = load_embeds_cpy.copy()
        try:
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
            
            # turn labels into integers
            labels = [int(label) for label in labels]
            
            clfCVscore_super, SVCscore_super, clfCVscore_sub, SVCscore_sub = linear_clf_embeddings_pca(embed_transformed, labels)
            # save the scores for each threshold in a dataframe
            scores = scores.append({'threshold': round(threshold, 6), 'num_features': load_embeds.shape[1], 'clfCV_super': clfCVscore_super, 'SVC_super': SVCscore_super, 'clfCV_sub': clfCVscore_sub, 'SVC_sub': SVCscore_sub}, ignore_index=True)
            
            # The following code is used to plot the PCA but is commented out for now
            
            # #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'pink', 'brown', 'grey', 'olive', 'cyan', 'tan', 'lime', 'teal', 'lavender', 'turquoise', 'darkgreen', 'gold', 'darkred', 'darkblue', 'darkorange', 'darkgrey', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkyellow', 'darkkhaki', 'darkolivegreen', 'darkorchid', 'darkseagreen', 'darkslateblue', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'greenyellow']
            # colors = ['b', 'g', 'r', 'c', 'm']
            # color = []
            # parent_labels = np.unique([int(np.unique(label)) // 100 for label in labels])
            # unique_labels = np.unique(labels)
            # for label in labels:
            #     # make the labels belong to their parent class, where labels 101-110 belong to class 1, 201-210 to class 2, etc.
            #     # label = int(unique_labels)
            #     parent_label = int(label) // 100
            #     parent_label_idx = np.where(parent_labels == parent_label)
            #     color.append(colors[parent_label_idx[0][0]])
                
            # #f, axarr = plt.subplots(2, 3)
            # # plt.figure(figsize=(10,8))
            # # plt.tight_layout()
            # # plt.title("PCA of the embeddings with variance thresholds")

            # if dim == 2:
            #     ax = axs[i//3, i%3]
            #     ax.scatter(embed_transformed[:, 0], embed_transformed[:, 1], color=color)
            #     ax.set_title(f"Variance threshold={threshold}")
            #     # plot the embeddings 2D with label colors for the 50 classes
            #     #plt.subplot(2, 3, i+1)
            #     #plt.suptitle('Threshold = ' + str(thresholds[i]))
            #     #plt.scatter(embed_transformed[:, 0], embed_transformed[:, 1], color=color)
            #     #plt.xlabel('PC1')
            #     #plt.ylabel('PC2')
            #     # if i in [0, 1, 2]:
            #     #     axarr[0, i].set_title('Threshold = ' + str(thresholds[i]))
            #     #     axarr[0, i].scatter(embed_transformed[:, 0], embed_transformed[:, 1], color=color)
            #     # elif i in [3, 4, 5]:
            #     #     axarr[1, i-3].set_title('Threshold = ' + str(thresholds[i]))
            #     #     axarr[1, i-3].scatter(embed_transformed[:, 0], embed_transformed[:, 1], color=color)

            # elif dim == 3:
            #     # plot the embeddings 3D with label colors for the 50 classes
            #     ax = fig.add_subplot(2, 3, i+1, projection='3d')
            #     ax.scatter(embed_transformed[:, 0], embed_transformed[:, 1], embed_transformed[:, 2], color=color)
            #     ax.set_title(f"Variance threshold={threshold}")
            #     #plt.subplot(1, 2, 1)
            #     # plt.title('PCA of the embeddings')
            #     # ax = plt.axes(projection='3d')
            #     # ax.scatter3D(embed_transformed[:, 0], embed_transformed[:, 1], embed_transformed[:, 2], color=color)
            #     # plt.xlabel('PC1')
            #     # plt.ylabel('PC2')
            
        except:
            return scores
        
    # plt.tight_layout()
    # plt.show()
    
    return scores
    
def linear_clf_embeddings_pca(embeddings, labels):
    """Classify the embeddings using a linear classifier.
    
    Args:
        embeddings (str): Embeddings fitted by PCA
        labels (str): Labels of the embeddings        
    Returns:
        floats: Accuracies of the classifiers.
    """
    embeds = embeddings
    
    from sklearn.model_selection import train_test_split
    train_embeds, test_embeds, train_labels, test_labels = train_test_split(embeds, labels, test_size=0.2, random_state=42)
    
    from sklearn.linear_model import LogisticRegressionCV
    super_train_labels = [int(label) // 100 for label in train_labels]
    
    clfCV_super = LogisticRegressionCV(cv=5, random_state=0, max_iter=500).fit(train_embeds, super_train_labels)
    clfCV_sub = LogisticRegressionCV(cv=5, random_state=0, max_iter=500).fit(train_embeds, train_labels)
    
    super_test_labels = [int(label) // 100 for label in test_labels]
        
    clfCVscore_super = clfCV_super.score(test_embeds, super_test_labels)
    clfCVscore_sub = clfCV_sub.score(test_embeds, test_labels)
    
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    SVCclf_super = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5))
    SVCclf_sub = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5))
    SVCclf_super.fit(train_embeds, super_train_labels)
    SVCclf_sub.fit(train_embeds, train_labels)
    SVCscore_super = SVCclf_super.score(test_embeds, super_test_labels)
    SVCscore_sub = SVCclf_sub.score(test_embeds, test_labels)
    
    # return scores
    return clfCVscore_super, SVCscore_super, clfCVscore_sub, SVCscore_sub
    
def linear_clf_embedding(embeddings, mode = "superclass"):
    """Function that takes the embeddings and trains a linear classifier on them.
       This should be done in a way so that the classifier seperates different classes
       based on their latent embedding.
       
       Args:
        embeddings (str): Embeddings fitted by PCA
        mode (str): Classification labels to use. Must be either 'superclass' or 'subclass'.  
       Returns:
        floats: Accuracies of the classifiers.
    """
    
    # load the embeddings
    load_embeds, labels = [], []
    max_embed_len = max(np.unique([len(np.load(embedding)) for embedding in embeddings], return_counts=True)[0])
    for embedding in embeddings:
        embed = np.load(embedding, allow_pickle=True)
        if mode == "superclass":
            label = int(embedding.split('/')[-2][0])
        elif mode == "subclass":
            label = int(embedding.split('/')[-2][:3])
        else:
                print("Invalid classification mode. Must be either 'superclass' or 'subclass'.")
                return
        labels.append(label)
        load_embeds.append(embed)
        
    labels = np.array(labels)
    embeds = np.array(load_embeds)
    
    from sklearn.model_selection import train_test_split
    train_embeds, test_embeds, train_labels, test_labels = train_test_split(embeds, labels, test_size=0.2, random_state=42)
    
    from sklearn.linear_model import LogisticRegressionCV
    clfCV = LogisticRegressionCV(cv=5, random_state=0).fit(train_embeds, train_labels)
    
    clfCVscore = clfCV.score(test_embeds, test_labels)
    
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    SVCclf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5))
    SVCclf.fit(train_embeds, train_labels)
    SVCscore = SVCclf.score(test_embeds, test_labels)
    
    print(f"{mode} classification")
    print(tabulate([['Logistic Regression CV', clfCVscore], ['SVC', SVCscore]], 
                   headers=['Classifier', 'Accuracy'], tablefmt='latex'))
    
    return clfCVscore, SVCscore
    
    
def plot_pca_scores(csv_path = 'output/embeddings/PCA_scores2.csv'):
    """Function should plot the different classifier scores for different thresholds.

    Args:
        csv_path (str, optional): Path to the csv file. Defaults to 'output/embeddings/PCA_scores_0_2.csv'.
    """
    df = pd.read_csv(csv_path)
    
    # plot the scores for all the thresholds
    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    plt.plot(df['threshold'], df['clfCV_super'], label='Logistic Regression CV (Super)', color='orange', linestyle='-')
    plt.plot(df['threshold'], df['SVC_super'], label='SVC (Super)', color = 'indigo', linestyle='-')
    plt.plot(df['threshold'], df['clfCV_sub'], label='Logistic Regression CV (Sub)', color='orange', linestyle='--')
    plt.plot(df['threshold'], df['SVC_sub'], label='SVC (Sub)', color = 'indigo', linestyle='--')
    
    plt.xlabel('Variance Threshold')
    plt.ylabel('Classifier Accuracy')
    plt.legend(title='Classification Algorithm')
    plt.savefig('output/embeddings/PCA_scores2.png')

if __name__ == "__main__":
    
    data_path = 'output/delta/embeddings/ECS50/'
    embeddings = []
    labels = []
    # get all the embeddings
    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
        embeddings.append([data_path + folder + '/' + file for file in os.listdir(data_path + folder) if file.endswith('.npy')])
        labels.append([file.split('-')[0] for file in os.listdir(data_path + folder) if file.endswith('.npy')])
        
    # flatten the lists
    labels = [item for sublist in labels for item in sublist]
    embeddings = [item for sublist in embeddings for item in sublist]
    
    # dim = 64 to compare with VAE
    dim = 64
    
    scores = embeddingPCA_manual(embeddings, dim)
    # save the scores to csv and plot the scores
    scores.to_csv('output/embeddings/PCA_scores_1_1.csv')
    plot_pca_scores('output/embeddings/PCA_scores_1_1.csv')
    
    # do linear classification on the embeddings
    linear_clf_embedding(embeddings, mode = "superclass")
    linear_clf_embedding(embeddings, mode = "subclass")