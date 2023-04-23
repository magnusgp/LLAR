LLAR
==============================

Learning audio representations for search and retrieval using

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
