import os
import json
import numpy as np
import pickle
import wandb
from tabulate import tabulate
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import information_retrieval.metrics as ir
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from another import VAE, Sampling
from auto_utils import plot_reconstructions, plot_label_clusters, visualize_conv_layers, save_embeddings, comparisons, tabulate_results, plot_test_reconstructions

def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths

def trainCV(specs, latent_space_dim, lr, batch_size, epochs, runname, beta_warmup):
    # reshape data and create KFold object
    specs = specs.reshape(specs.shape[0], 256, 256, 1)
    kfold = KFold(n_splits=5, shuffle=True)
    losses_over_folds = pd.DataFrame(columns=["Loss", "Reconstruction Loss", "KL Loss"])
    
    # set up final validation set
    val_data = specs[-100:]
    
    # update specs to not contain val_data
    specs = specs[:-100]
    
    print(f"Training VAE with {specs.shape[0]} training examples and {val_data.shape[0]} validation examples...")
    
    # build autoencoder
    vae = VAE(latent_dim=latent_space_dim, beta_warmup=beta_warmup)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss = vae.total_loss_tracker)
    vae.encoder.summary()
    vae.decoder.summary()
    fold_no = 1
    for train, test in kfold.split(specs):
        print(f'\nTraining for fold {fold_no}...')
        vae.train(specs[train], epochs, batch_size, runname, fold_no)
        scores = vae.evaluate(specs[test])
        print(f'\nValidation scores for fold {fold_no}: Loss: {round(scores[0],3)}; Reconstruction Loss: {round(scores[1],3)}; KL Loss: {round(scores[2],3)}')
        losses_over_folds.loc[fold_no] = scores
        folder = "runs/" + runname + "/CV/fold_" + str(fold_no) + "/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        vae.save(folder)
        fold_no += 1
        
    val_scores = vae.evaluate(val_data)
    plot_test_reconstructions(vae, val_data, mode = "pixel")
    print(f'\nFinal validation scores: Loss: {round(val_scores[0],3)}; Reconstruction Loss: {round(val_scores[1],3)}; KL Loss: {round(val_scores[2],3)}')
    losses_over_folds.to_csv("runs/" + runname + "/losses_over_folds.csv")
    
    return vae

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    
    # set random seed for tensorflow
    tf.random.set_seed(42)
    
    config = json.load(open("config_new.json"))
    
    run = wandb.init(project="autoencoder", config=config)
    
    # create the run directory if it doesn't exist
    if not os.path.exists("uns/" + config["run_name"]):
        os.makedirs("runs/" + config["run_name"])
        
    # if the run directory already exists, don't overwrite it. instead, create a new run directory
    # the new directory should be the same as the old one, but with a number appended to the end
    if os.path.exists("runs/" + config["run_name"]):
        os.makedirs("runs/" + config["run_name"] + "_COPY")
        # change the run name to the new run name
        config["run_name"] = config["run_name"] + "_COPY"
    
    if not os.path.exists("logs/training/" + config["run_name"]):
        os.makedirs("logs/training/" + config["run_name"])
    
    with open("runs/" + config["run_name"] + "/config_" + config["run_name"]  + ".json", "w") as f:
        json.dump(config, f)
    
    print(config["spectrograms_path"])
    specs, file_paths = load_fsdd(config["spectrograms_path"])
    
    vae = trainCV(specs, config["latent_space_dim"], config["learning_rate"], 
                         config["batch_size"], config["epochs"], 
                         config["run_name"], config["beta_warmup"])
    
    savefolder = "runs/" + config["run_name"]
    
    vae.save(savefolder)
    
    if not os.path.exists(savefolder + "/embeddings/"):
        os.makedirs(savefolder + "/embeddings/")
        
    if not os.path.exists(savefolder + "/comparisons/"):
        os.makedirs(savefolder + "/comparisons/")
    
    save_embeddings(vae, specs, file_paths, savefolder + "/embeddings/")
    
    plot_reconstructions(specs, file_paths, vae, savefolder)
    
    labels = []
    for i in range(len(file_paths)):
        labels.append(int(file_paths[i].split('/')[-1][:14][:1]))

    plot_label_clusters(vae, specs, labels, savefolder)
    
    y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred = comparisons(savefolder + "/embeddings/", labels, savefolder + "/comparisons/")
    
    k = 5
    print(tabulate_results(euc_acc, cos_acc, labels, y_true, euc_pred, cos_pred, k, savefolder))
    
    # if not os.path.exists("runs/" + config["run_name"] + "/layer_viz/"):
    #     os.makedirs("runs/" + config["run_name"] + "/layer_viz/")
    # savepath = "runs/" + config["run_name"] + "/layer_viz/"
    
    # # visualize encoder layers
    # encoder_layers = [layer for layer in vae.encoder.layers]
    # for layer in encoder_layers:
    #     if "conv" in layer.name:
    #         visualize_conv_layers(vae.encoder, specs[0], layer.name, savepath)
    
    # decoder_layers = [layer for layer in vae.decoder.layers]
    # for layer in decoder_layers:
    #     if "conv" in layer.name:
    #         visualize_conv_layers(vae.decoder, specs[0], layer.name, savepath)
