# LLAR
==============================

### Learning Latent Audio Representations for search and retrieval using supervised and unsupervised learning methods

==============================

#### This repository was prepared as a part of the Bachelor thesis project at the Technical University of Denmark for the study program Artificial Intelligence and Data Science.

==============================

In order to use this repository for recreation of results, first download the publicly Environmental Sound Classification 50 (ESC-50) dataset here: <https://github.com/karolpiczak/ESC-50>

Then, clone the repository and paste the dataset into the 'data/ESC50'-folder or choose your own structure.

```
git clone https://github.com/magnusgp/LLAR.git
```

Make sure that you are logged into wandb.ai, otherwise you will get an error when logging model progress and visualizations.
However, this logging can be removed manually from the code if it is not desired.

==============================

Run the Makefile to setup the environment or use your own. The dependencies are listed in requirements.txt

Depending on the model you want to use, cd into either "autoencoder" or "yamnet". Please note that these are only short start-up guides and does not cover the entire codebase. You are more than welcome to explore the scripts and I will update this page soon.

==============================

## Autoencoder

```
cd autoencoder
```
Configure the training of the model in the config_new.json file. Then, run the training script.
Please be aware of the hyperparameters set and that we are runnung a 5-fold CV where the weights are saved for each run.

```
python3 -m train_new.py
```
After model training, explore the embeddings generated and do similarity-based search and retrieval!

You can also load a pretrained model to do so (just change the run directory in the script load_model.py):
```
python3 -m load_model.py
```

==============================

## YAMNet

```
cd autoencoder
```
Generate the embeddings using the pretrained YAMNet model:
```
python3 -m embeddings.py
```
Then, do similarity-based search and retrieval:
```
python3 -m comparisons.py
```
==============================

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile for initializing the project
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── ESC50          <- Data folder. If you use this repository for replication of results, 
    │                        download the ESC-50 dataset and paste it here
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── autoencoder        <- Source code and relevant folders for the variational autoencoder used in the project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── fsdd           <- Folder containing preprocessed data
    │   │   └── pixel_values.png <- Pixel value likelihood distribution
    │   │
    │   ├── logs         <- Model checkpoints and training logs
    │   │
    │   ├── wandb        <- Wandb.ai logfiles used for tracking model training
    │   │
    │   ├── runs         <- Saving folder for model weigths
    │   │
    │   ├── output       <- Visualizations and comparisons
    │   │
    │   └── source files  <- source files
    │
    ├── yamnet             <- Source code and relevant folders for using YAMNet in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Folder containing data
    │   │
    │   ├── logs         <- Model checkpoints and training logs
    │   │
    │   ├── wandb        <- Wandb.ai logfiles used for tracking model training
    │   │
    │   ├── runs         <- Saving folder for model weigths
    │   │
    │   ├── output       <- Visualizations and comparisons
    │   │
    │   └── source files  <- source files
    │   │
    ├── information_retrieval  <- Source code for IR scoring and comparison.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── comparison_mcnemar.py <- Source file for McNemar test
    │   │
    │   └── information_retrieval.py  <- Source file for all IR scoring
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
