"""Attempt of implementation of a variational autoencoder with skip connections."""

# fix imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import regularizers
from keras.layers import Activation
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import Adam

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

import os
import pickle
import datetime
import numpy as np
import tensorflow as tf


class VAESKIP:
    """Implements a variational autoencoder with skip connections.

    This class implements a variational autoencoder with skip connections
    as described in [1]_.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    hidden_dim : int
        Dimensionality of the hidden layer.
    latent_dim : int
        Dimensionality of the latent layer.
    n_layers : int
        Number of hidden layers.
    activation : str
        Activation function to use.
    dropout : float
        Dropout rate to use.
    batch_norm : bool
        Whether to use batch normalization.
    batch_size : int
        Batch size to use.
    learning_rate : float
        Learning rate to use.
    n_epochs : int
        Number of epochs to train for.
    random_state : int
        Random state to use.
    verbose : bool
        Whether to print progress.

    Attributes
    ----------
    encoder_ : keras.models.Model
        Encoder model.
    decoder_ : keras.models.Model
        Decoder model.
    autoencoder_ : keras.models.Model
        Autoencoder model.
    history_ : dict
        Training history.

    References
    ----------
    .. [1] https://arxiv.org/abs/1606.08921
    """

    def __init__(self, input_dim, hidden_dim=512, latent_dim=128, n_layers=7,
                 activation='relu', dropout=0.2, skip_connections=True, 
                 validation_split=0.1, batch_norm=True, batch_size=128, 
                 learning_rate=0.001, n_epochs=100, random_state=None, 
                 verbose=True):
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose
        
        self._build_autoencoder()
        
    def summary(self):
        """Prints a summary of the model."""
        self.autoencoder_.summary()
        
    def glu_activation(self, x):
        """Gated linear unit activation function."""
        return x * Activation('sigmoid')(x)
    
    def reconstruction_loss(self, y_true, y_pred):
        """Reconstruction loss."""
        return -1/2 * (tf.math.log(tf.linalg.det(self.sigma)) + tf.math.log(2*np.pi) + tf.matmul(tf.matmul((y_pred - self.mu), tf.linalg.inv(self.sigma)), tf.transpose(y_pred - self.mu)))
    
    def kl_loss(self, y_true, y_pred):
        """KL divergence loss."""
        return -1/2 * (1 + self.z_log_var - tf.math.square(self.z_mean) - tf.math.exp(self.z_log_var))
    
    def loss(self, y_true, y_pred):
        """Total loss."""
        return self.reconstruction_loss(y_true, y_pred) + self.kl_loss(y_true, y_pred)
        
    def _build_autoencoder(self):
        """Builds the autoencoder model."""
        # Encoder part
        inputs = Input(shape=self.input_dim)
        x = inputs 
        x = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x)
        x3 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x3)
        x2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x2)
        x1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x1)
        x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x)
        x = Flatten()(x)
        
        # Bottleneck part
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Lambda(self._sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        self.encoder_ = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder part
        x = z
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x)
        Add()([x, x1])
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x)
        Add()([x, x2])
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x)
        Add()([x, x3])
        x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x)
        x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
        x = self.glu_activation(x)
        outputs = Dense(self.input_dim, activation='sigmoid')(x)
        self.decoder_ = Model(inputs, outputs, name='decoder')
        
        # Autoencoder
        outputs = self.decoder_(self.encoder_(inputs)[2])
        self.autoencoder_ = Model(inputs, outputs, name='autoencoder')
        self.autoencoder_.compile(optimizer=Adam(lr=self.learning_rate),
                                  loss=self.loss)
        self.autoencoder_.summary()
        return self.autoencoder_
    
    def _sampling(self, args):
        """Samples from the latent space."""
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def train(self, X, y=None):
        """Train the model to fit to the data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Training labels.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X)
        self._build_autoencoder()
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = "logs/training/run5/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=40*self.batch_size)
        self.model.save_weights(checkpoint_path.format(epoch=0))
        self.history_ = self.autoencoder_.fit(X, X,
                                                batch_size=self.batch_size,
                                                epochs=self.n_epochs,
                                                verbose=self.verbose,
                                                valudation_split=self.validation_split,
                                                callbacks=[tensorboard_callback, cp_callback],
                                                shuffle=True).history
        return self
    
    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAESKIP(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder
    
    def save(self, filepath):
        """Saves the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to the file.
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self._save_parameters(filepath)
        self._save_weights(filepath)
    
    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)
    
    