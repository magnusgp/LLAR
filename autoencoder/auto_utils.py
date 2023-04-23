
"""
## Display visualizations
"""

import os

import wandb

import numpy as np
from tqdm import tqdm

import librosa

from tabulate import tabulate

import matplotlib.pyplot as plt

import information_retrieval.metrics as ir

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score

from sklearn.decomposition import PCA
    
def plot_reconstructions(specs, file_paths, vae, savefolder):
    # plot 5 reconstructed images and their original images
    num_recons = 5
    # generate 5 random indices
    idxs = np.random.randint(0, len(specs), num_recons)
    plt.figure(figsize=(16, 8))
    plt.title("Original Images, Latent Representations, and Reconstructed Images")
    plt.tight_layout()
    for i in range(num_recons):
        current_img = specs[idxs[i]]
        # add batch dim to current_img
        current_img = current_img[np.newaxis, ...]
        latent_representation = vae.encoder.predict(current_img)
        reconstructed_image = vae.decoder.predict(latent_representation[2])[0][0, ...]
        wandb.log({"original_image": [wandb.Image(current_img[0, ...])], 
                    "reconstructed_image (means)": [wandb.Image(reconstructed_image)]})
        # remove batch dim again to show imgs
        current_img = current_img[0, ...]
        # display original
        ax = plt.subplot(2, num_recons, i + 1)
        # title: original image
        ax.set_title("{}".format(file_paths[idxs[i]].split('/')[-1][:14]))
        plt.imshow(specs[idxs[i]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # # display latent representation
        # ax = plt.subplot(3, 5, i + 1 + 5)
        # # title: original image
        # if i == 2:
        #     ax.set_title("Latent Representation")
        # # the latent representation is a vector with [latent_dim] numbers, plot them
        # latent_concat = np.concatenate((latent_representation[0], latent_representation[1]), axis=1)
        # plt.plot(latent_concat[0])
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, num_recons, i + 1 + num_recons)
        if i == num_recons // 2:
            ax.set_title("Reconstructed")
        plt.imshow(reconstructed_image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    savepath = os.path.join(savefolder, "reconstructions.png")
    print("Saving reconstructions to {}".format(savepath))
    plt.savefig(savepath)
    

"""
## Display how the latent space clusters different classes
"""

def plot_label_clusters(vae, data, labels, savefolder):
    # display a 2D plot of the classes in the latent space by doing PCA
    z_mean, _, _ = vae.encoder.predict(data)
    
    # do PCA
    pca = PCA(n_components=2)
    z_mean = pca.fit_transform(z_mean)
    
    # plot the PCA-reduced data
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("PC1 of z_mean")
    plt.ylabel("PC2 of z_mean")
    savepath = os.path.join(savefolder, "latent_space.png")
    plt.savefig(savepath)
    
def plot_pixel_distribution(specpath = 'autoencoder/fsdd/spectograms_big', savefolder = 'autoencoder/output'):
    """ Plot the distribution of pixel values in the dataset. """
    specs = []
    for file in os.listdir(specpath):
        filepath = os.path.join(specpath, file)
        specs.append(np.load(filepath))
    specs = np.array(specs)
    plt.figure(figsize=(12, 8))
    plt.hist(specs.flatten(), bins=50)
    plt.xlabel("Pixel value")
    plt.ylabel("Number of pixels")
    plt.savefig(savefolder + '/pixel_distribution_bins.png')
    
def plot_test_reconstructions(vae, data, mode="pixel", savefolder = 'autoencoder/output'):
    """ Plot reconstructions. If the mode is pixel, plot the distribution of the pixel values. 
        Else plot the reconstructed images. """
    latent_representation, reconstructed_images = [], []    
    
    if mode == "pixel":
        plt.figure(figsize=(12, 8))
        plt.title("Pixel distribution of the reconstructed images (test set)")
        plt.tight_layout()
        
        for i in range(len(data)):
            latent_representation.append(vae.encoder.predict(data[i][np.newaxis, ...]))
            reconstructed_images.append(vae.decoder.predict(latent_representation[-1][-1])[0][0, ...])
            wandb.log({"original_image": [wandb.Image(data[i][np.newaxis, ...][0, ...])], 
                       "reconstructed_image (means)": [wandb.Image(reconstructed_images[-1])]})
        
        plt.hist(np.array(reconstructed_images).flatten(), bins=50)
        print(f"Saving to {savefolder + 'pixel_distribution_reconstructed.png'}")
        plt.savefig(savefolder + 'pixel_distribution_reconstructed.png')
        
        wandb.log({"pixel_distribution_reconstructed": [wandb.Image(plt)]})
        
    elif mode == "normaldist":
        plt.figure(figsize=(12, 8))
        # plot a normal distribution with mean 0 and std 1
        plt.hist(np.random.normal(0, 1, size=100000), bins=50)
        plt.savefig(savefolder + 'normal_distribution.png')
        
    else:
        raise ValueError("Mode must be either 'pixel' or 'normaldist'.")
    
def visualize_conv_layers(model, spec, layer_name, savefolder):

    layer_output=model.get_layer(layer_name).output  #get the Output of the Layer

    intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about

    intermediate_prediction=intermediate_model.predict(spec.reshape(1,256,256,1)) #predicting in the Intermediate Node

    row_size=5
    col_size=5

    img_index=0

    print(np.shape(intermediate_prediction))
    #---------------We will subplot the Output of the layer which will be the layer_name----------------------------------#

    fig, ax=plt.subplots(row_size,col_size,figsize=(10,8)) 
    plt.title("Visualization of the Convolutional Layer "+layer_name)
    for row in range(0,row_size):
        for col in range(0,col_size):
            if row == 0:
                if col == 2:
                    ax[row][col].imshow(spec.reshape(256, 256))
                    ax[row][col].axis('off')
                    ax[row][col].set_title("Original Image")
                    row += 1
                else:
                    ax[row][col].axis('off')
                    continue
            else:
                ax[row][col].imshow(intermediate_prediction[0, :, :, img_index])
                ax[row][col].axis('off')
                ax[row][col].set_title(layer_name+"_"+str(img_index+1))
            img_index=img_index+1 #Increment the Index number of img_index variable
    plt.show()
    savepath = os.path.join(savefolder, "conv_layer" + layer_name + ".png")
    plt.savefig(savepath)
    
def save_embeddings(vae, data, file_paths, savefolder):
    for file, data in tqdm(zip(file_paths, data), total=len(file_paths)):
        current_img = data
        current_img = current_img[np.newaxis, ...]
        current_z_mean, current_z_log_var, _ = vae.encoder.predict(current_img, verbose=0)
        # save the latent representation in a file
        #embedding = np.concatenate((current_z_mean, current_z_log_var), axis=1)
        embedding = current_z_mean
        savepath = os.path.join(savefolder, file.split('/')[-1].split('.')[0] + ".npy")
        np.save(savepath, embedding)
        
def VAESimilarity(embedding, embeddings, labels, file_paths, idx):
    """ Function that calculates the euclidean distance between 
        the input vector and the embedded vectors.

    Args:
        embedding (str): The file name of the input vector
        embeddings (list): A list of file names of the embedded vectors
        labels (list): A list of labels corresponding to the embedded vectors

    Returns:
        min_distance (str): The file name of the most similar vector
    """
    input_vector = np.load(embedding, allow_pickle=True)
    input_name = embeddings[idx].split('/')[-1].split('.')[0]
    embedded_vectors = {}
    for i, embedding in enumerate(embeddings):
        if input_name in embeddings[i]:
            continue
        else:
            embedded_vectors[embeddings[i]] = np.load(embedding, allow_pickle=True)
    
    eucdistances, cosdistances = {}, {}
    shapes = {}
    count = 0
    # calculate the euclidean distance between the input vector and the embedded vectors
    for file_name, vector in embedded_vectors.items():
        if file_name.split('/')[-1].split('.')[0] in input_name:
            continue
        else:
            # if the input vector is the first vector, initialize the dictionary
            input_vector = np.squeeze(np.asarray(input_vector))
            vector = np.squeeze(np.asarray(vector))
            
            if file_name == list(embedded_vectors.keys())[0]:
                #eucdistances = {file_name: np.linalg.norm(input_vector - vector)}
                eucdistances = {file_name: np.linalg.norm(input_vector - vector)}
                cosdistances = {file_name: 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))}
            # otherwise add the distance to the dictionary
            else:
                #eucdistances[file_name] = np.linalg.norm(input_vector - vector)
                eucdistances[file_name] = np.linalg.norm(input_vector - vector)
                cosdistances[file_name] = 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))
            
    # find the minimum distance
    min_eucdistance = min(eucdistances, key=eucdistances.get)
    
    min_cosdistance = min(cosdistances, key=cosdistances.get)
    
    # return the filename of the minimum distance and the euclidean distance
    return min_eucdistance, eucdistances, min_cosdistance, cosdistances
        

def comparisons(data_path, labels, savefolder, top_k = 5):
    """Function that compares all embeddings to each other and plots a matrix that shows how often
    the most similar embeddings beling to the same class/folder as the original embedding.

    Args:
        data_path (str, optional): Path of the embeddings.
    """
    y_true, euc_pred, cos_pred = np.array([]), np.array([]), np.array([])
    euc_top_k_preds, cos_top_k_preds = [], []
    
    embeddings = []
    # get all the embeddings
    for file in os.listdir(data_path):
        embeddings.append(data_path + file)
    
    for i, file in tqdm(enumerate(os.listdir(data_path)), total = len(os.listdir(data_path))):
        if file.endswith('.npy') and not file.endswith('_trimmed.npy'):
            # load the current embedding
            current_embedding = str(data_path + file)
            y_true = np.append(y_true, labels[i])
            # compare the current embedding to all other embeddings
            euc_filename, euc_distances, cos_filename, cos_distances = VAESimilarity(current_embedding, embeddings, labels, data_path, i)
            euc_pred = np.append(euc_pred, int(euc_filename.split('/')[-1][:3]))
            cos_pred = np.append(cos_pred, int(cos_filename.split('/')[-1][:3]))
            
            euc_top_k_preds.append([int([key for key, val in euc_distances.items() if val == searchval][0].split('/')[-1][:3]) for searchval in np.sort(np.array(list(euc_distances.values())))[:top_k]])
            cos_top_k_preds.append([int([key for key, val in cos_distances.items() if val == searchval][0].split('/')[-1][:3]) for searchval in np.sort(np.array(list(cos_distances.values())))[:top_k]])

                
    # get the indices of the misclassified embeddings
    euc_idxs = ((euc_pred == y_true) == False).nonzero()
    cos_idxs = ((cos_pred == y_true) == False).nonzero()
                
    euc_conf = confusion_matrix(y_true, euc_pred)
    cos_conf = confusion_matrix(y_true, cos_pred)
    
    euc_acc = accuracy_score(y_true, euc_pred)
    cos_acc = accuracy_score(y_true, cos_pred)    
    
    # save the results to .npy files
    
    ars = [y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds]
    arnames = ["y_true", "euc_conf", "cos_conf", "euc_idxs", "cos_idxs", "labels", "euc_acc", "cos_acc", "euc_pred", "cos_pred", "euc_top_k_preds", "cos_top_k_preds"]

    for i, ar in enumerate(ars):
        np.save(savefolder + arnames[i] + '.npy', ar)
    
    return y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds

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
        pass
    elif mode == "baseline_superclass":
        # make random predictions of 1-5
        labels = [int(str(label)[:1]) for label in labels]
        y_true = [int(str(label)[:1]) for label in y_true]
        euc_pred = [np.random.randint(1,6) for i in range(len(euc_pred))]
        cos_pred = [np.random.randint(1,6) for i in range(len(cos_pred))]
        euc_acc = ir.Accuracy(labels = labels, y_true = y_true, pred = euc_pred)
        cos_acc = ir.Accuracy(labels = labels, y_true = y_true, pred = cos_pred)
        cos_top_k_preds = [[np.random.randint(1,6) for i in range(k)] for j in range(len(cos_top_k_preds))]
        euc_top_k_preds = [[np.random.randint(1,6) for i in range(k)] for j in range(len(euc_top_k_preds))]
    
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
    
    with open(savefolder + '/' + str(mode) + '_results.txt', 'w') as f:
        f.write(tab)
        f.write(tab2)
        
    return tab, tab2

def visualize_five_waveforms(path = "data/ECS50/"):
    """Function that visualizes one waveform from each superclass (100-500).
    """
    plt.figure(figsize=(16, 12))
    plt.tight_layout()
    for i in range(1,6):
        idx = np.random.randint(0, 10)
        # each folder is named like this: 101 - Dog, 102 - Rooster, etc.
        for folder in os.listdir(path):
                # if the folder is not a directory, continue
            if not os.path.isdir(path + folder):
                continue
            if int(folder[0]) == i:
                if int(folder[2:3]) == idx:
                    for file in os.listdir(path + folder):
                        if file.endswith('.wav') and not file.endswith('_trimmed.wav'):
                            signal, sr = librosa.load(path + folder + '/' + file)
                            plt.subplot(5, 1, i)
                            plt.title(folder)
                            plt.plot(signal)
                            # remove x axis
                            plt.xticks([])
                            plt.ylabel('Amplitude')
                            break
    plt.savefig('waveforms.png')
    
def plot_waveform_and_spectrogram(waveform = "101 - Dog", wavpath = "data/ECS50/", specpath = "autoencoder/fsdd/spectograms_big/"):
    """Function that plots a waveform and its corresponding log mel spectrogram.
       The spectrogram is already computed in specpath and just needs to be loaded.

    Args:
        waveform (str, optional): Name of the waveform. Defaults to "101 - Dog".
        wavpath (str, optional): Path to the waveform parent directories. Defaults to "data/ECS50/".
        specpath (str, optional): Path to the spectrogram corresponding to the waveform. Defaults to "autoencoder/fsdd/spectograms_big/".
    """
    plt.figure(figsize=(16, 12))
    plt.tight_layout()
    for folder in os.listdir(wavpath):
        # if the folder is not a directory, continue
        if not os.path.isdir(wavpath + folder):
            continue
        if folder == waveform:
            for file in os.listdir(wavpath + folder):
                if file.endswith('.wav') and not file.endswith('_trimmed.wav'):
                    id = file.split('-')[1]
                    take = file.split('-')[-1].split('.')[0]
                    signal, sr = librosa.load(wavpath + folder + '/' + file)
                    plt.subplot(2, 1, 1)
                    plt.title("Raw waveform of file: " + folder + '/' + file)
                    plt.plot(signal)
                    # remove x axis
                    plt.xticks([])
                    plt.ylabel('Amplitude')
                    break
            for file in os.listdir(specpath):
                if file.endswith('.npy') and not file.endswith('_trimmed.npy'):
                    if file.split('-')[0] == folder[0:3] and file.split('-')[2] == id and file.split('-')[-1].split('.')[0] == take:
                        spec = np.load(specpath + file)
                        plt.subplot(2, 1, 2)
                        plt.title("Corresponding Log Mel Spectrogram")
                        plt.imshow(spec, aspect='auto', origin='lower')
                        plt.ylabel('Frequency')
                        plt.xlabel('Time')
                        plt.tight_layout()
                        break
        plt.savefig(waveform + '.png')
    
                    
if __name__ == "__main__":
    #plot_waveform_and_spectrogram(waveform = "510 - Hand saw")
    #plot_pixel_distribution()
    pass