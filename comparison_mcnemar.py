import numpy as np
import tabulate

from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_vae_yamnet(y_true_yamnet, y_true_vae, pred_yamnet, pred_vae, mode = "euclidean", classtype = "super"):
    if classtype == "super":
        y_true_yamnet = np.array([int(str(label)[:1]) for label in y_true_yamnet])
        y_true_vae = np.array([int(str(label)[:1]) for label in y_true_vae])
        pred_yamnet = np.array([int(str(label)[:1]) for label in pred_yamnet])
        pred_vae = np.array([int(str(label)[:1]) for label in pred_vae])
    elif classtype == "sub":
        pass
    else:
        print(f"\nClasstype {classtype} not supported. Use 'super' or 'sub'.")

    # Create contingency table from the predictions.
    a = np.sum(np.logical_and(pred_yamnet == y_true_yamnet, pred_vae == y_true_vae))
    b = np.sum(np.logical_and(pred_yamnet != y_true_yamnet, pred_vae == y_true_vae))
    c = np.sum(np.logical_and(pred_yamnet == y_true_yamnet, pred_vae != y_true_vae))
    d = np.sum(np.logical_and(pred_yamnet != y_true_yamnet, pred_vae != y_true_vae))

    # Create contingency table
    contingency_table = np.array([[a, b], [c, d]])
    
    print(f"Contingency table for the McNemar test with {classtype}class labels ({mode} measurement):")
    print(tabulate.tabulate([
                             ["VAE Correct", a, b],
                            ["VAE Incorrect", c, d]], headers=[" ", "YAMNet Correct", "YAMNet Incorrect"], 
                            tablefmt="latex"))
    
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')
    
    return result.pvalue


if __name__ == "__main__":
    y_true_yamnet = np.load("output/mean/comparisons/y_true.npy")
    y_true_vae = np.load("autoencoder/runs/run_final_latent_dim_mini_batch_128/comparisons/y_true.npy")
    
    mode = "cosine"
    classtype = "sub"
    
    if mode == "euclidean":
        pred_yamnet = np.load("output/mean/comparisons/euc_pred.npy")
        pred_vae = np.load("autoencoder/runs/run_final_latent_dim_mini_batch_128/comparisons/euc_pred.npy")
    elif mode == "cosine":
        pred_yamnet = np.load("output/mean/comparisons/cos_pred.npy")
        pred_vae = np.load("autoencoder/runs/run_final_latent_dim_mini_batch_128/comparisons/cos_pred.npy")
    else:
        print("Mode not recognized. Please choose between 'euclidean' and 'cosine'.")
        
    mcnemar_vae_yamnet(y_true_yamnet, y_true_vae, pred_yamnet, pred_vae, mode, classtype)
    