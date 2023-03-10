
"""
Creates all embeddings for a subsample of the data files to do similarity analysis on.
    
"""

from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import os
from tqdm import tqdm

import params as yamnet_params
import yamnet as yamnet_model


def embeddings(data_path, data_type="normal"):
    
    assert data_path, 'Usage: embeddings.py <data path>'
    # load the model
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    # we don't need the class names for this as we are not doing inference
    # yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')
    filedir = os.listdir(data_path)
    
    if data_type == "normal":
        try:
            processed = np.load('data/ECS50/processed.npy', allow_pickle=True)
        except:
            processed = []
    
        for folder in tqdm(filedir):
            print(f'Processing folder: {folder}')
            
            # wierd mac thing, but we're not using it anyways
            if folder == '.DS_Store':
                continue
            
            # check if the folder has already been processed
            # if folder in processed:
            #     print(f"Folder {folder} has already been processed, skipping...")
            #     continue
            
            # loop over the files in the folder
            try:
                files = os.listdir(data_path + folder)
            except:
                print(f'\nFolder/file {folder} is not a data folder, skipping...')
                continue
            files = [data_path + folder + '/' + file for file in files]
            
            for file_name in files:
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

                # Get YAMNet embeddings.
                _, embeddings, spectograms = yamnet(waveform)
                # save the embeddings to a file for later use in the output/embeddings folder
                if not os.path.exists('output/embeddings/ECS50/' + folder):
                    os.makedirs('output/embeddings/ECS50/' + folder)
                np.save('output/embeddings/ECS50/' + folder + '/' + file_name.split('/')[-1].split('.')[0] + '.npy', embeddings)
                if not os.path.exists('output/spectograms/ECS50/' + folder):
                    os.makedirs('output/spectograms/ECS50/' + folder)
                np.save('output/spectograms/ECS50/' + folder + '/' + file_name.split('/')[-1].split('.')[0] + '.npy', spectograms)
            # add the folder to the processed list
            processed = np.append(processed, folder)
        np.save('data/ECS50/processed.npy', processed)
        
    elif data_type == "trimmed":
        try:
            processed = np.load('data/ECS50/processed_trimmed.npy', allow_pickle=True)
        except:
            processed = []
    
        for folder in tqdm(filedir):
            print(f'Processing folder: {folder}')
            
            # wierd mac thing, but we're not using it anyways
            if folder == '.DS_Store':
                continue
            
            # check if the folder has already been processed
            if folder in processed:
                print(f"Folder {folder} has already been processed, skipping...")
                continue
            
            # loop over the files in the folder
            try:
                files = os.listdir(data_path + folder)
            except:
                print(f'\nFolder/file {folder} is not a data folder, skipping...')
                continue
            files = [data_path + folder + '/' + file for file in files if file.endswith('_trimmed.wav')]
            
            for file_name in files:
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

                # Get YAMNet embeddings.
                _, embeddings, spectograms = yamnet(waveform)
                # save the embeddings to a file for later use in the output/embeddings folder
                if not os.path.exists('output/embeddings/ECS50_trimmed/' + folder):
                    os.makedirs('output/embeddings/ECS50_trimmed/' + folder)
                np.save('output/embeddings/ECS50_trimmed/' + folder + '/' + file_name.split('/')[-1].split('.')[0] + '.npy', embeddings)
                if not os.path.exists('output/spectograms/ECS50_trimmed/' + folder):
                    os.makedirs('output/spectograms/ECS50_trimmed/' + folder)
                np.save('output/spectograms/ECS50_trimmed/' + folder + '/' + file_name.split('/')[-1].split('.')[0] + '.npy', spectograms)
            # add the folder to the processed list
            processed = np.append(processed, folder)
        np.save('data/ECS50/processed_trimmed.npy', processed)
    
if __name__ == '__main__':
    data_path = 'data/ECS50/'
    
    data_type = "normal"
    
    embeddings(data_path, data_type=data_type)
    
    if data_type == "normal":
        # get total number of embeddings in all folders
        c = 0
        for folder in os.listdir('output/embeddings/ECS50/'):
            try:
                os.listdir('output/embeddings/ECS50/' + folder)
            except:
                continue
            c += len(os.listdir('output/embeddings/ECS50/' + folder))
    elif data_type == "trimmed":
        # get total number of embeddings in all folders
        c = 0
        for folder in os.listdir('output/embeddings/ECS50_trimmed/'):
            try:
                os.listdir('output/embeddings/ECS50_trimmed/' + folder)
            except:
                continue
            c += len(os.listdir('output/embeddings/ECS50_trimmed/' + folder))
    
    print(f"Finished creating embeddings for all files in {data_path} with data type {data_type}\n")
    print(f"This has resulted in a total of {c} embeddings\n")