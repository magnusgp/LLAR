"""
Title: Variational AutoEncoder
Author: [Magnus Guldberg Pedersen]
Description: Convolutional Variational AutoEncoder (VAE) trained on ECS50 log mel spectrograms.
Accelerator: GPU
"""

"""
## Setup
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from tensorflow.keras import layers

import wandb
from wandb.keras import WandbCallback

"""
## Create the sampling layer
"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

"""
## Define the VAE as a `Model` with a custom `train_step` and `test_step`
"""



class VAE(keras.Model):
    def __init__(self, latent_dim, beta_warmup, **kwargs):
        super().__init__(**kwargs)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        self.verbosity_mode = 2
        
        self.latent_dim = latent_dim
        if beta_warmup == "True":
            self.beta = K.variable(value=0.0)
            self.beta_warmup = True
        else:
            self.beta_warmup = False
        
        self._build(latent_dim = latent_dim)
        
    def glu(self, x):
        return x * layers.Activation('sigmoid')(x)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
        
    def _build(self, latent_dim = 128):
        self._build_encoder(latent_dim)
        self._build_decoder(latent_dim)
        
    """
    ## Build the encoder
    """
    def _build_encoder(self, latent_dim = 128):
        encoder_inputs = keras.Input(shape=(256, 256, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(encoder_inputs)
        #x = layers.ReLU()(x)
        x = layers.Dense(32, activation=self.glu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Dense(64, activation=self.glu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Dense(128, activation=self.glu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Dense(256, activation=self.glu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder = encoder
        return encoder

    """
    ## Build the decoder
    """
    def _build_decoder(self, latent_dim = 128):
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(32 * 32 * 256, activation="relu")(latent_inputs)
        x = layers.Reshape((32, 32, 256))(x)
        x = layers.Conv2DTranspose(256, 3, activation="relu", strides=1, padding="same")(x)
        x = layers.Dense(256, activation=self.glu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Dense(128, activation=self.glu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Dense(64, activation=self.glu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Dense(32, activation=self.glu)(x)
        x = layers.BatchNormalization()(x)
        decoder_mean = layers.Conv2DTranspose(1, 1, activation="sigmoid", padding="same")(x)
        decoder_log_var = layers.Conv2DTranspose(1, 1, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, [decoder_mean, decoder_log_var], name="decoder")
        self.decoder = decoder
        return decoder
    
    # define callback to change the value of beta at each epoch
    def warmup(self, epoch):
        value = (epoch/10.0) * (epoch <= 10.0) + 1.0 * (epoch > 10.0)
        print("beta:", value)
        K.set_value(self.beta, value)
        
    def do_nothing(self, batch):
        k = batch
        
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)[0]
            var_reconstruction = self.decoder(z)[1]
            error = data - reconstruction
            
            # # using GMMs, doesn't work
            # pi = 0.5
            # zero_var = 0.01
            
            # var_reconstruction = tf.maximum(var_reconstruction, 1e-8)  # Avoid division by zero
            # log_likelihood = -0.5 * tf.reduce_sum(tf.square(error) / var_reconstruction + tf.math.log(var_reconstruction) + np.log(2 * np.pi), axis=[1, 2, 3])
            # zero_likelihood = tf.reduce_sum(tf.math.log(pi) - np.log(np.sqrt(2 * np.pi * zero_var)) - tf.square(data) / (2 * zero_var), axis=[1, 2, 3])
            # total_log_likelihood = tf.reduce_logsumexp(tf.stack([log_likelihood, zero_likelihood], axis=1), axis=1)
            # reconstruction_loss = -tf.reduce_mean(total_log_likelihood)
            
            # correct loss
            reconstruction_loss = tf.reduce_sum(tf.square(error)/(2 * tf.square(var_reconstruction)), axis=[1, 2, 3])
            
            # old loss without variance
            #reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(error), axis=[1, 2, 3]))
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            if self.beta_warmup:
                total_loss = self.beta * reconstruction_loss + kl_loss
            else:
                total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def train(self, specs, epochs, batch_size, runname, fold_no):
        log_dir = "logs/fit/" + runname + "_fold_" + str(fold_no) + "/"
        checkpoint_path = "logs/training/"+runname+"/cp-{epoch:04d}.ckpt"

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        wandbCallback = WandbCallback()
        
        if self.beta_warmup:
            wu_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, log: self.warmup(epoch))
            cbs = [tensorboard_callback, wandbCallback, wu_cb]
        else:
            cbs = [tensorboard_callback, wandbCallback]

        self.fit(specs,
                epochs=epochs, 
                validation_data=specs,
                batch_size=batch_size,
                callbacks=cbs,
                verbose=self.verbosity_mode)
        
    def test_step(self, data):
        y = data
        #output mean values of pixels instead of sample itself for prediction
        y_pred = self.decoder(self.encoder(data)[0])
        #y_pred = self.decoder(self.encoder(data)[2])
        
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
        
    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        #self._save_parameters(save_folder)
        self._save_weights(save_folder)
        
    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    def _save_parameters(self, save_folder):
        parameters = [
            self.encoder,
            self.decoder,
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
            
    def _save_weights(self, save_folder):
        decoder_save_path = os.path.join(save_folder, "decoder_weights.h5")
        encoder_save_path = os.path.join(save_folder, "encoder_weights.h5")
        self.encoder.save_weights(encoder_save_path)
        self.decoder.save_weights(decoder_save_path)
        
    @classmethod
    def load(cls, latent_dim = 128, beta_warmup = "False", load_folder="."):
        autoencoder = VAE(latent_dim=latent_dim, beta_warmup=beta_warmup)
        encoder_path = os.path.join(load_folder, "encoder_weights.h5")
        decoder_path = os.path.join(load_folder, "decoder_weights.h5")
        autoencoder.encoder.load_weights(encoder_path)
        autoencoder.decoder.load_weights(decoder_path)
        return autoencoder

if __name__ == "__main__":
    autoencoder = VAE(latent_dim=128, beta_warmup=False)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam())
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
