
"""This file does similarity measures between an 
input vector and the rest of the embedded vectors

The vectors are the outputs of the YAMNet model

The similarity measures will be done using simple euclidean distance
"""

import numpy as np
import os
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import librosa

def similarity(input_vector, embeddings, plot="plot", embed_mean=True):
    """Returns the euclidean distance between the input vector and the embedded vectors

    Args:
        input_vector (str: path): Path to the input vector to compare to the embedded vectors.
        embeddings (str: path): Path to the folder containing the embedded vectors.
        plot (str): Argument whether to plot embedding shape distribution or not. Plotting is on by default.
        embed_mean (bool): Argument whether to take the mean of the embedded vectors or not. This is done by 
        default as the vector embeddings otherwise would have very few other vectors with the same shape,
        causing them to be skipped in the search or throw an error.

    Returns:
        min_eucdistance (str): The file name of the most similar vector with regards to euclidian distance
        eucdistances[min_eucdistances] (float): The euclidean distance between the input vector and the 
                                                most similar vector
                                                
        min_cosdistance (str): The file name of the most similar vector with regards to cosine similarity
        cosdistances[min_cosdistance] (float): The cosine similarity between the input vector and the 
                                                most similar vector
    """
    # load the input vector
    input_name = input_vector.split('/')[-1].split('.')[0]
    input_full_name = input_vector
    input_vector = np.load(input_vector)
    # load the embedded vectors from the folder into a dictionary with their filenames as keys
    for i, file_name in tqdm(enumerate(embeddings)):
        # skip the input vector
        if file_name == input_vector:
            continue
        else:
            if i == 0:
                embedded_vectors = {file_name: np.load(file_name)}
            else:
                embedded_vectors[file_name] = np.load(file_name)
    
    eucdistances, cosdistances, shapes = {}, {}, {}
    count = 0
    if embed_mean:
        input_vector = np.mean(input_vector, axis=0)
    # calculate the euclidean distance between the input vector and the embedded vectors
    for file_name, vector in tqdm(embedded_vectors.items()):
        # if the keyword embed_mean is True, take the mean of the embedded vectors to make shapes match
        if embed_mean:
            vector = np.mean(vector, axis=0)
        # log the shape of the vector
        if np.shape(vector) in shapes.keys():
            # increment the occurence of the shape
            shapes[np.shape(vector)] += 1
        else:
            shapes[np.shape(vector)] = 1
        # skip the input vector
        if file_name.split('/')[-1].split('.')[0] in input_name:
            continue
        # shapes must match to perform the similarity measure
        elif np.shape(input_vector) == np.shape(vector):
            # if the input vector is the first vector, initialize the dictionary
            input_vector = np.squeeze(np.asarray(input_vector))
            vector = np.squeeze(np.asarray(vector))
            if file_name == list(embedded_vectors.keys())[0]:
                eucdistances = {file_name: np.linalg.norm(input_vector - vector)}
                cosdistances = {file_name: 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))}
            # otherwise add the distance to the dictionary
            else:
                eucdistances[file_name] = np.linalg.norm(input_vector - vector)
                cosdistances[file_name] = 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))
        else: 
            #print('The input vector and the embedded vector {} do not have the same shape'.format(file_name))
            # skip the vector if it does not have the same shape as the input vector and increment skip counter
            count += 1
            
    # find the minimum distance
    min_eucdistance = min(eucdistances, key=eucdistances.get)
    min_cosdistance = min(cosdistances, key=cosdistances.get)
    
    # for file_name, distance in distances.items():
        # print(f'\nThe distance between the input vector and {file_name} is {distance}')
        
    pctskipped = round(count / len(embedded_vectors) * 100, 2)
    print(f'\n{count} vectors (out of {len(embedded_vectors)} vectors) were skipped due to shape mismatch')
    print(f'\nThis means that {pctskipped} % of the vectors were skipped')
    if pctskipped > 50:
        print('Yikes, that\'s a lot of vectors skipped')
        
    # print the most common shapes of the vectors in sorted order with their amount of occurences
    # print(f'\nThe most common shapes of the vectors are: {sorted(shapes.items(), key=lambda x: x[1], reverse=True)}')
    # plot the most common shapes of the vectors
    if plot == "plot":
        commonshapes = sorted(shapes.items(), key=lambda x: x[1], reverse=True)
        plt.figure(figsize=(10, 7))
        plt.bar([str(commonshapes[i][0]) for i, shape in enumerate(commonshapes)], [commonshapes[i][1] for i, shape in enumerate(commonshapes)])
        # rotate the x-axis labels
        plt.xticks(rotation=45)
        plt.show()
    
    # return the filename of the minimum distance and the euclidean distance
    #return min_eucdistance, eucdistances[min_eucdistance], min_cosdistance, cosdistances[min_cosdistance]
    return min_eucdistance, eucdistances[min_eucdistance], min_cosdistance, cosdistances[min_cosdistance]

def ECS_similarity(input_vector, embeddings, plot="plot", embed_mean=True):
    """Returns the euclidean distance between the input vector and the embedded vectors
       Has the exact same functionality as the similarity function above, but just supports data handling for
       the ECS-50 dataset instead. As of now there is no difference, but this might occur when the embeddings 
       are created. 

    Args:
        input_vector (str: path): Path to the input vector to compare to the embedded vectors.
        embeddings (str: path): Path to the folder containing the embedded vectors.
        plot (str): Argument whether to plot embedding shape distribution or not. Plotting is on by default.
        embed_mean (bool): Argument whether to take the mean of the embedded vectors or not. This is done by 
        default as the vector embeddings otherwise would have very few other vectors with the same shape,
        causing them to be skipped in the search or throw an error.

    Returns:
        min_eucdistance (str): The file name of the most similar vector with regards to euclidian distance
        eucdistances[min_eucdistances] (float): The euclidean distance between the input vector and the 
                                                most similar vector
                                                
        min_cosdistance (str): The file name of the most similar vector with regards to cosine similarity
        cosdistances[min_cosdistance] (float): The cosine similarity between the input vector and the 
                                                most similar vector
    """
    # load the input vector
    input_name = input_vector.split('/')[-1].split('.')[0]
    input_full_name = input_vector
    input_vector = np.load(input_vector)
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
    if embed_mean:
        input_vector = np.mean(input_vector, axis=0)
    # calculate the euclidean distance between the input vector and the embedded vectors
    for file_name, vector in embedded_vectors.items():
        # if the keyword embed_mean is True, take the mean of the embedded vectors to make shapes match
        if embed_mean:
            vector = np.mean(vector, axis=0)
        # log the shape of the vector
        if np.shape(vector) in shapes.keys():
            # increment the occurence of the shape
            shapes[np.shape(vector)] += 1
        else:
            shapes[np.shape(vector)] = 1
        # skip the input vector
        if file_name.split('/')[-1].split('.')[0] in input_name:
            continue
        # shapes must match to perform the similarity measure
        elif np.shape(input_vector) == np.shape(vector):
            # if the input vector is the first vector, initialize the dictionary
            input_vector = np.squeeze(np.asarray(input_vector))
            vector = np.squeeze(np.asarray(vector))
            if file_name == list(embedded_vectors.keys())[0]:
                eucdistances = {file_name: np.linalg.norm(input_vector - vector)}
                cosdistances = {file_name: 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))}
            # otherwise add the distance to the dictionary
            else:
                eucdistances[file_name] = np.linalg.norm(input_vector - vector)
                cosdistances[file_name] = 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))
        else: 
            # skip the vector if it does not have the same shape as the input vector and increment skip counter
            count += 1
            
    # find the minimum distance
    min_eucdistance = min(eucdistances, key=eucdistances.get)
    
    min_cosdistance = min(cosdistances, key=cosdistances.get)
        
    # pctskipped = round(count / len(embedded_vectors) * 100, 2)
    # print(f'\n{count} vectors (out of {len(embedded_vectors)} vectors) were skipped due to shape mismatch')
    # print(f'\nThis means that {pctskipped} % of the vectors were skipped')
    
    # plot the most common shapes of the vectors
    if plot == "plot":
        commonshapes = sorted(shapes.items(), key=lambda x: x[1], reverse=True)
        plt.figure(figsize=(10, 7))
        plt.bar([str(commonshapes[i][0]) for i, shape in enumerate(commonshapes)], [commonshapes[i][1] for i, shape in enumerate(commonshapes)])
        # rotate the x-axis labels
        plt.xticks(rotation=45)
        plt.show()
    
    # return the filename of the minimum distance and the euclidean distance
    return min_eucdistance, eucdistances[min_eucdistance], min_cosdistance, cosdistances[min_cosdistance]

def mean_delta_similarity(input_vector, embeddings, plot="plot", data = "mean", VAE = False):
    """Same as the above function, but instead takes the mean, variance and delta features of the 
       embedded vectors and uses the concatenated mean, variance and delta features as the embedded
       vector to compare to the input vector.

    Args:
        input_vector (str: path): Path to the input vector to compare to the embedded vectors.
        embeddings (str: path): Path to the folder containing the embedded vectors.
        plot (str): Argument whether to plot embedding shape distribution or not. Plotting is on by default.
        embed_mean (bool): Argument whether to take the mean of the embedded vectors or not. This is done by 
        default as the vector embeddings otherwise would have very few other vectors with the same shape,
        causing them to be skipped in the search or throw an error.
        data (str): Argument whether to use the mean, variance or delta features of the embedded vectors.
        VAE (bool): Argument whether the embedded vectors are from a VAE or not.

    Returns:
        min_eucdistance (str): The file name of the most similar vector with regards to euclidian distance
        eucdistances[min_eucdistances] (float): The euclidean distance between the input vector and the 
                                                most similar vector
                                                
        min_cosdistance (str): The file name of the most similar vector with regards to cosine similarity
        cosdistances[min_cosdistance] (float): The cosine similarity between the input vector and the 
                                                most similar vector
    """
    # load the input vector
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
    
    # create all root dirs
    if os.path.exists("output/" + ' '.join(data) + "/embeddings/" + input_full_name.split('/')[-2]) == False:
        os.makedirs("output/" + ' '.join(data) + "/embeddings/" + input_full_name.split('/')[-2])
    
    # save the input_vector as the embedding
    np.save("output/" + ' '.join(data) + "/embeddings/" + input_full_name.split('/')[-2] + '/' + input_full_name.split('/')[-1], input_vector)
    
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
        
        # mean = np.mean(vector, axis=0)
        # var = np.var(vector, axis=0)
        # delta = np.mean(librosa.feature.delta(vector), axis=0)
        
        # vector = np.concatenate((mean, var, delta), axis=0)
        
        vector = np.concatenate(vec, axis=0)

        if file_name.split('/')[-1].split('.')[0] in input_name:
            continue
        
        # shapes must match to perform the similarity measure
        elif np.shape(input_vector) == np.shape(vector):
            # if the input vector is the first vector, initialize the dictionary
            input_vector = np.squeeze(np.asarray(input_vector))
            vector = np.squeeze(np.asarray(vector))
            
            if file_name == list(embedded_vectors.keys())[0]:
                eucdistances = {file_name: np.linalg.norm(input_vector - vector)}
                cosdistances = {file_name: 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))}
            # otherwise add the distance to the dictionary
            else:
                eucdistances[file_name] = np.linalg.norm(input_vector - vector)
                cosdistances[file_name] = 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))
        else: 
            # skip the vector if it does not have the same shape as the input vector and increment skip counter
            count += 1
            
    # find the minimum distance
    min_eucdistance = min(eucdistances, key=eucdistances.get)
    
    min_cosdistance = min(cosdistances, key=cosdistances.get)
    
    # plot the most common shapes of the vectors
    if plot == "plot":
        commonshapes = sorted(shapes.items(), key=lambda x: x[1], reverse=True)
        plt.figure(figsize=(10, 7))
        plt.bar([str(commonshapes[i][0]) for i, shape in enumerate(commonshapes)], [commonshapes[i][1] for i, shape in enumerate(commonshapes)])
        # rotate the x-axis labels
        plt.xticks(rotation=45)
        plt.show()
    
    # return the filename of the minimum distance and the euclidean distance
    return min_eucdistance, eucdistances, min_cosdistance, cosdistances

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
    input_vector = embedding
    input_name = file_paths[idx].split('/')[-1].split('.')[0]
    embedded_vectors = {}
    for i, embedding in enumerate(embeddings):
        if input_name in file_paths[i]:
            continue
        else:
            embedded_vectors[file_paths[i]] = embedding
    
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
                eucdistances = {file_name: np.linalg.norm(input_vector - vector)}
                cosdistances = {file_name: 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))}
            # otherwise add the distance to the dictionary
            else:
                eucdistances[file_name] = np.linalg.norm(input_vector - vector)
                cosdistances[file_name] = 1 - (np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector)))
            
    # find the minimum distance
    min_eucdistance = min(eucdistances, key=eucdistances.get)
    
    min_cosdistance = min(cosdistances, key=cosdistances.get)
    
    # return the filename of the minimum distance and the euclidean distance
    return min_eucdistance, eucdistances[min_eucdistance], min_cosdistance, cosdistances[min_cosdistance]

if __name__ == '__main__':
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    
    # call sript by python similarity.py <plot_option> <dataset_toption>
    
    try:
        plot = sys.argv[1]
    except:
        print(f"\n Plotting tool not set. Continuing without plotting similarities.")
        plot = ""
    try:
        dataset = sys.argv[2]
    except:
        print(f"\n Unknown or incorrect dataset. Proceeding with the standard ECS50 dataset")
        dataset = "ECS50"
    # import the input vector as a string from the command line
    try:
        input_vector = input('Enter the path to the input vector: ')
        assert input_vector.endswith('.npy'), 'The input vector must be a .npy file'
        assert os.path.exists(input_vector), 'The input vector does not exist'
    except:
        print('\nThe input vector does not exist or is not a .npy file. Using the default input vector instead')
        input_vector = 'output/embeddings/p226_169_mic1.npy' if dataset == "VCTK" else 'output/embeddings/ECS50/402 - Mouse click/5-223317-A.npy'
        print(f"\n Selected input vector: {input_vector}")

    # get the most similar vector and log to console
    if dataset == "VCTK":
        # get all VCTK embeddings in the embeddings folder
        embeddings = ['output/embeddings/' + file for file in os.listdir('output/embeddings')]
        
        # remove embeddings that end with _mic2.npy for now
        for embedding in embeddings:
            if embedding.endswith('_mic2.npy'):
                embeddings.remove(embedding)
                
        most_sim_euc, euc_distance, most_sim_cos, cos_distance = similarity(input_vector, embeddings, plot)

    elif dataset == "ECS50":
        # get all ECS50 embeddings in the embeddings folder
        embeddings = ['output/embeddings/ECS50/' + file for file in os.listdir('output/embeddings/ECS50')]
        
        most_sim_euc, euc_distance, most_sim_cos, cos_distance = ECS_similarity(input_vector, embeddings, plot)
    
    print(f"\nMost similar vector (Euclidian Distance): {most_sim_euc} with a distance of {euc_distance}")
    print(f"\nMost similar vector (Cosine Similarity): {most_sim_cos} with a distance of {cos_distance}")
    print(f"\nInput vector: {input_vector}")