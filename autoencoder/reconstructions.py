import autoencoder as ae
from train import load_fsdd
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_reconstructions(specs, file_paths, latent_representations, reconstructed_images):
    # plot 5 reconstructed images and their original images
    # generate 5 random indices
    idxs = np.random.randint(0, len(specs), 5)
    plt.figure(figsize=(12, 8))
    plt.title("Original Images, Latent Representations, and Reconstructed Images")
    plt.tight_layout()
    for i in range(5):
        # display original
        ax = plt.subplot(3, 5, i + 1)
        # title: original image
        ax.set_title("{}".format(file_paths[idxs[i]].split('/')[-1][:14]))
        plt.imshow(specs[idxs[i]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display latent representation
        ax = plt.subplot(3, 5, i + 1 + 5)
        # title: original image
        if i == 2:
            ax.set_title("Latent Representation")
        # the latent representation is a vector with [latent_dim] numbers, plot them
        plt.plot(latent_representations[idxs[i]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, 5, i + 1 + 10)
        if i == 2:
            ax.set_title("Reconstructed")
        plt.imshow(reconstructed_images[idxs[i]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("runs/run8/reconstructed_glu.png")

if __name__ == "__main__":
    run_path = "autoencoder/runs/run27"
    vae = ae.VAE.load(run_path)
    # load the config file from the run
    with open(run_path + "/config.json", "r") as f:
        config = json.load(f)
    specs, file_paths = load_fsdd("autoencoder/fsdd/spectograms_big")
    latent_representations = vae.encoder.predict(specs)
    reconstructed_images = vae.decoder.predict(latent_representations)
    vae._calculate_reconstruction_loss(specs, reconstructed_images)
    plot_reconstructions(specs, file_paths, latent_representations, reconstructed_images)
