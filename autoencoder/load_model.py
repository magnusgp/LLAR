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

from train_new import load_fsdd

from another import VAE, Sampling
from auto_utils import plot_reconstructions, plot_label_clusters, visualize_conv_layers, save_embeddings, comparisons, tabulate_results, plot_test_reconstructions



def loadVAE(specs, run_no, latent_space_dim, lr, batch_size, epochs, runname, beta_warmup):

    # build autoencoder
    vae = VAE(latent_dim=latent_space_dim, beta_warmup=beta_warmup)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss = vae.total_loss_tracker)
    
    print(f"Loading saved model...")
    
    if run_no == "final_latent_dim_256":
        run_no = "final_latent_dim_256/CV/fold_1"
    
    vae.encoder.load_weights('autoencoder/runs/run_'+str(run_no)+'/encoder_weights.h5')
    vae.decoder.load_weights('autoencoder/runs/run_'+str(run_no)+'/decoder_weights.h5')
        
    return vae

def plot_loaded_model(specs, file_paths, vae, savefolder):
    specs = specs.reshape(specs.shape[0], 256, 256, 1)
    
    val_data = specs[-100:]
    
    plot_test_reconstructions(vae, val_data, mode = "pixel", savefolder = savefolder)
    
    plot_reconstructions(val_data, file_paths, vae, savefolder)
    

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    
    # set random seed for tensorflow
    tf.random.set_seed(42)
    
    runs = ["5_32"]
    
    for run_no in runs:

        config = json.load(open("autoencoder/runs/run_"+str(run_no)+"/config_run_"+str(run_no)+".json"))
        
        if config["spectrograms_path"] == "fsdd/spectograms_big":
            config["spectrograms_path"] = "autoencoder/fsdd/spectograms_big"
        
        run = wandb.init(project="autoencoder", name="run_"+str(run_no), config=config)
        
        # create the run directory if it doesn't exist
        if not os.path.exists("autoencoder/runs/" + config["run_name"]):
            os.makedirs("autoencoder/runs/" + config["run_name"])
        
        if not os.path.exists("logs/training/" + config["run_name"]):
            os.makedirs("logs/training/" + config["run_name"])
        
        with open("autoencoder/runs/" + config["run_name"] + "/config_" + config["run_name"]  + ".json", "w") as f:
            json.dump(config, f)
        
        print(config["spectrograms_path"])
        specs, file_paths = load_fsdd(config["spectrograms_path"])
        
        vae = loadVAE(specs, run_no, config["latent_space_dim"], config["learning_rate"], 
                            config["batch_size"], config["epochs"], 
                            config["run_name"], config["beta_warmup"])
        
        savefolder = "autoencoder/runs/run_"+str(run_no)
        
        if not os.path.exists(savefolder + "/embeddings/"):
            os.makedirs(savefolder + "/embeddings/")
            
        if not os.path.exists(savefolder + "/comparisons/"):
            os.makedirs(savefolder + "/comparisons/")
            
        # if there's no saved embeddings, save them
        # also check for embeddings with wrong dim (happens sometimes are HPC run)
        try:
            if np.load(savefolder + "/embeddings/" + os.listdir(savefolder + "/embeddings/")[0]).flatten().shape[0] != config["latent_space_dim"]:
                print(f"Correct embeddings not found, computing them now ...")
                save_embeddings(vae, specs, file_paths, savefolder + "/embeddings/")
        except:
            save_embeddings(vae, specs, file_paths, savefolder + "/embeddings/")
        
        labels = []
        for i in range(len(file_paths)):
            labels.append(int(file_paths[i].split('/')[-1][:14][:3]))
            
        top_k = 5
        
        try:
            # load all of these numpy arrays: y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds
            # raise Exception("Load comparisons")
            y_true = np.load(savefolder + "/comparisons/y_true.npy")
            euc_conf = np.load(savefolder + "/comparisons/euc_conf.npy")
            cos_conf = np.load(savefolder + "/comparisons/cos_conf.npy")
            euc_idxs = np.load(savefolder + "/comparisons/euc_idxs.npy")
            cos_idxs = np.load(savefolder + "/comparisons/cos_idxs.npy")
            labels = np.load(savefolder + "/comparisons/labels.npy")
            euc_acc = np.load(savefolder + "/comparisons/euc_acc.npy")
            cos_acc = np.load(savefolder + "/comparisons/cos_acc.npy")
            euc_pred = np.load(savefolder + "/comparisons/euc_pred.npy")
            cos_pred = np.load(savefolder + "/comparisons/cos_pred.npy")
            euc_top_k_preds = np.load(savefolder + "/comparisons/euc_top_k_preds.npy")
            cos_top_k_preds = np.load(savefolder + "/comparisons/cos_top_k_preds.npy")
            
            assert len(euc_top_k_preds[0]) >= top_k, "there are fewer than top_k predictions... \n reproducing files with new top_k value. \n"
            
            print("Loaded saved comparisons.")
                
        except:
            print("No saved comparisons found. Computing comparisons...")
        
            y_true, euc_conf, cos_conf, euc_idxs, cos_idxs, labels, euc_acc, cos_acc, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds = comparisons(savefolder + "/embeddings/", labels, savefolder + "/comparisons/", top_k=top_k)
        
        plot_loaded_model(specs, file_paths, vae, savefolder)
        
        print(f"\n Evaluating model with latent dimension: {config['latent_space_dim']}")
        
        print(f"\nResults using top k: {top_k}")
        
        tab1, tab2 = tabulate_results(euc_acc, cos_acc, labels, y_true, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds, top_k, savefolder, mode = "superclass")
        
        print(tab1)
        print("\n\n")
        print(tab2)
        
        tab3, tab4 = tabulate_results(euc_acc, cos_acc, labels, y_true, euc_pred, cos_pred, euc_top_k_preds, cos_top_k_preds, top_k, savefolder, mode = "subclass")
        print(tab3)
        print("\n\n")
        print(tab4)