""" Main script for the project.
    
    This defines the overall pipeline of the project.
    It contains the main function that calls all the other functions.
"""

import embeddings as emb
import comparisons as comp
import visualizationECS as vis
import PCA as pca

def main():
    # create embeddings
    emb.embeddings()
    
    # compare embeddings
    comp.comparisons()
    
    # perform PCA
    pca.embeddingPCA()
    
    # plot comparison with closest waveform
    vis.yamnetplot()
    
if __name__ == "__main__":
    main()