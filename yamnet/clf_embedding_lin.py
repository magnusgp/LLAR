import os
import numpy as np
from tabulate import tabulate


def linear_clf_embeddings(data_path, mode="superclass"):
    """Classify the embeddings using a linear classifier.
       The embeddings are a path to a folder where .npy matrices are stored.
       These embeddings should be loaded and then used to train a linear classifier.
       Should return the accuracy of the classifier.

    Args:
        embeddings (str): Path to the embeddings folder.
        mode (str, optional): Classification mode, either "superclass" or "subclass". Defaults to "superclass".
        
    Returns:
        float: Accuracy of the classifier.
    """
    labels = []
    
    embeddings = []
    # get all the embeddings
    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
        embeddings.append(['output/embeddings/ECS50/' + folder + '/' + file for file in os.listdir('output/embeddings/ECS50/' + folder)])
        
    # flatten the list
    embeddings = [item for sublist in embeddings for item in sublist]
    embeds = []
    for embedding in embeddings:
        if embedding.endswith(".npy"):
            if mode == "superclass":
                label = int(embedding.split('/')[-2][0])
            elif mode == "subclass":
                label = int(embedding.split('/')[-2][:3])
            else:
                print("Invalid classification mode. Must be either 'superclass' or 'subclass'.")
                return
            labels.append(label)
            embeds.append(np.load(embedding).flatten())
    labels = np.array(labels)
    
    embeds = np.array(embeds)
    
    embeds_2d = np.vstack(embeds)
    
    # make sure that the embeds array have dim (2000, 10240)
    embeds = embeds_2d.reshape(2000, 10240)
    
    from sklearn.model_selection import train_test_split
    train_embeds, test_embeds, train_labels, test_labels = train_test_split(embeds, labels, test_size=0.2, random_state=42)
    
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, BayesianRidge
    clfCV = LogisticRegressionCV(cv=5, random_state=0).fit(train_embeds, train_labels)
    
    clfCVscore = clfCV.score(test_embeds, test_labels)
    
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    SVCclf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5))
    SVCclf.fit(train_embeds, train_labels)
    SVCscore = SVCclf.score(test_embeds, test_labels)
    
    # return accuracy
    return clfCVscore, SVCscore

if __name__ == "__main__":
    clfCVscore, SVCscore = linear_clf_embeddings("output/embeddings/ECS50", mode = "superclass")

    clfCVscore_sub, SVCscore_sub = linear_clf_embeddings("output/embeddings/ECS50", mode = "superclass")
    
    print(f"Classification accuracy for YAMNET:\n")
    print(tabulate([["Superclasses", clfCVscore, SVCscore], ["Subclasses", clfCVscore_sub, SVCscore_sub]], headers=["", "Logistic Regression", "Logistic Regression CV", "Lasso", "Bayesian Ridge", "SVC"]))         
    print("\n")
