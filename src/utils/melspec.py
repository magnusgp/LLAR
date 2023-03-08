"""
Creates all mel spectograms for a subsample of the data files to do similarity analysis on.
    
"""

from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import os

import params as yamnet_params
import yamnet as yamnet_model


def melspec(data_path):
    assert data_path, 'Usage: melspec.py <data path>'
    # load the model
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    # we don't need the class names for this
    # yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')
    files = os.listdir(data_path)
    # add the path to the file name
    files = [data_path + file for file in files]
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

        # Predict YAMNet classes.
        _, _, melspec = yamnet(waveform)
        # save the embeddings to a file for later use in the output/embeddings folder
        np.save('output/spectograms/' + file_name.split('/')[-1].split('.')[0] + '.npy', melspec)
    
if __name__ == '__main__':
    melspec('data/')