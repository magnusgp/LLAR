""" Trim silence from the wav files in order to represent embeddings more accurately.
    This is done by using the librosa.effects.trim function.
"""

import os
import librosa
# import tqdm
import soundfile as sf
import numpy as np
from tqdm import tqdm
import resampy

import embeddings as emb

def trim_wav(data_path):
    """Function that trims the silence from the wav files.

    Args:
        data_path (str): Path of the raw audio files.
    """
    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
    
    for i, folder in tqdm(enumerate(os.listdir(data_path)), total = len(os.listdir(data_path))):
        if folder == '.DS_Store':
            continue
        # check if the folder is a directory
        if not os.path.isdir(data_path + '/' + folder):
            continue
        files = os.listdir(data_path + '/' + folder)
        # add the path to the file name
        files = [data_path + '/' + folder + '/' + file for file in files if file.endswith('.wav')]
        for file_name in files:
            if file_name.endswith('.DS_Store'):
                continue
            if file_name.endswith('_trimmed.wav'):
                continue
            # load the wav file
            y, sr = librosa.load(file_name)
            # trim the wav file
            yt, index = librosa.effects.trim(y)
            # create a new file name, with the trimmed suffix
            trim_name = file_name.split(".")[0] + "_trimmed.wav"
            # use soundfile to write the trimmed wav file
            sf.write(trim_name, yt, sr)
    print("Trimming done!")
    
if __name__ == "__main__":
    data_path = 'data/ECS50'
    trim_wav(data_path)