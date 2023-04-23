# Visualize the spectograms computed by the melspec.py script (in the output/spectograms folder)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import resampy
import soundfile as sf
import tensorflow as tf

import params as yamnet_params
import yamnet as yamnet_model
import similarity as sim

def spectograms():
    """Function that plots the spectograms directly from the data path.
    """
    files = os.listdir('output/spectograms')
    files = [file for file in files if file.endswith('.npy')]

    for file in files:
        spectogram = np.load('output/spectograms/' + file)
        plt.imshow(spectogram)
        plt.show()
        
def waveforms(data_path='data/'):
    """Function that plots the waveforms of the data files

    Args:
        data_path (str, optional): Path of the raw audio files. Defaults to 'data/'.
    """
    files = os.listdir(data_path)
    # add the path to the file name
    files = [data_path + file for file in files]
    for file_name in files[0]:
        # Decode the WAV file.
        wav_data, sr = sf.read(file_name, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        plt.plot(waveform)
        plt.title(file_name)
        plt.show()
        
def yamnetplot(data):
    """Function that does similarity seaching using YAMNet on the input vectors.
    Plots both the waveform and the log mel spectogram of the input vector and 
    the most similar vector found in the data path.

    Args:
        data (array of strings): Array containing strings with filenames that should
        be used as an input vector. Be careful not to load a lot of files here, as it
        plots for every file.
    """
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5', by_name=True)
    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')
    
    if len(data) > 25:
        print(f"\n Caution! You have selected a lot of files. This may take a while.")
    
    for file_name in data:
        # Decode the WAV file.
        wav_data, sr = sf.read(file_name, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != params.sample_rate:
            waveform = resampy.resample(waveform, sr, params.sample_rate)

        # Predict YAMNet classes.
        scores, embeddings, spectrogram = yamnet(waveform)
        
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        infered_class = yamnet_classes[scores_np.mean(axis=0).argmax()]
        
        # find the embedding corresponding to the file
        # input_vector = [file for file in os.listdir('output/embeddings/ECS50') if file_name.split('/')[-1].split('.')[0] in file][0]
        # print(f"\nInput vector: {input_vector}\n")
        # # add the path to the file name
        # input_vector = 'output/embeddings/ECS50/' + input_vector
        
        # hard coding the input vector for now
        input_vector = 'output/embeddings/ECS50/307 - Laughing/1-1791-A.npy'
        
        embeddingfolders = ['output/embeddings/ECS50/' + folder for folder in os.listdir('output/embeddings/ECS50') if os.path.isdir('output/embeddings/ECS50/' + folder)]
        
        embeddings = [folder + '/' + file for folder in embeddingfolders for file in os.listdir(folder) if file.endswith('.npy') and not file.endswith('_trimmed.npy')]
    
        # remove embeddings that end with _mic2.npy for now
        for embedding in embeddings:
            if embedding.endswith('_mic2.npy'):
                embeddings.remove(embedding)
            elif embedding.endswith('_trimmed.npy'):
                embeddings.remove(embedding)

        # Plot the waveform of the most similar audio file
        most_sim, distance, most_sim_cos, cosdist = sim.ECS_similarity(input_vector, embeddings)
        
        # get the right name of the file (without the path and the .npy ending)
        
        most_sim = most_sim.split('/')[-1].split('.')[0]
        
        # get the first 4 characters of the file name
        #folderid = most_sim[:4]
        #most_sim = 'data/' + folderid + '/' + most_sim + '.flac'
        # loop over the files in the data folder and find the right file
        dir = 'data/ECS50'
        datapath = os.listdir('data/ECS50')
        for folder in datapath:
            # check if the folder is a directory
            if not os.path.isdir(dir + '/' + folder):
                continue
            foldername = folder
            folder = os.listdir(dir + '/' + folder)
            for file in folder:
                if most_sim in file:
                    # check if the file is a .wav file
                    if file.endswith('.wav') and not file.endswith('_trimmed.wav'):
                        # join the file paths to make the full path
                        most_sim = dir + '/' + foldername + '/' + file
                        #most_sim = os.path.join(datapath, folder, file)
                if most_sim_cos in file:
                    if file.endswith('.wav') and not file.endswith('_trimmed.wav'):
                        most_sim_cos = dir + '/' + foldername + '/' + file
        print(f"Searching in file: {most_sim}\n")
        # find the right file in the data folder
        #most_sim = [file for file in data if most_sim in file][0]
        
        # Decode the WAV file.
        wav_data_sim, sr = sf.read(most_sim, dtype=np.int16)
        assert wav_data_sim.dtype == np.int16, 'Bad sample type: %r' % wav_data_sim.dtype
        waveform_sim = wav_data_sim / 32768.0  # Convert to [-1.0, +1.0]
        waveform_sim = waveform_sim.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform_sim.shape) > 1:
            waveform_sim = np.mean(waveform_sim, axis=1)
        if sr != params.sample_rate:
            waveform_sim = resampy.resample(waveform_sim, sr, params.sample_rate)
            
        # Get similar waveform.
        _, _, spectrogram_sim = yamnet(waveform_sim)
        
        spectrogram_sim_np = spectrogram_sim.numpy()
        
        # make the figure dynamic size to fit the titles
        plt.figure(figsize=(10, 10))
        #plt.title("Infered class: " + infered_class)
        # Plot the waveform.
        plt.subplot(2, 2, 1)
        plt.plot(waveform)
        plt.title('Input file: ' + file_name)
        plt.xticks([])
        plt.yticks([])

        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(2, 2, 2)
        plt.title('Log Mel Spectrogram')
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
        plt.xticks([])
        plt.yticks([])
        
        # plot the waveform
        plt.subplot(2, 2, 3)
        plt.plot(waveform_sim)
        plt.title('Most similar file: ' + most_sim)
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(2, 2, 4)
        plt.title("Log Mel Spectogram")
        plt.imshow(spectrogram_sim_np.T, aspect='auto', interpolation='nearest', origin='lower')
        plt.xticks([])
        plt.yticks([])
        

        # # Plot and label the model output scores for the top-scoring classes.
        # mean_scores = np.mean(scores, axis=0)
        # top_n = 10
        # top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
        # plt.subplot(3, 1, 3)
        # plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

        # # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
        # # values from the model documentation
        # patch_padding = (0.025 / 2) / 0.01
        # plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
        # # Label the top_N classes.
        # yticks = range(0, top_n, 1)
        # plt.yticks(yticks, [yamnet_classes[top_class_indices[x]] for x in yticks])
        # _ = plt.ylim(-0.5 + np.array([top_n, 0]))
        
        plt.show()
        
if __name__ == '__main__':
    yamnetplot(['data/ECS50/307 - Laughing/1-1791-A.wav'])