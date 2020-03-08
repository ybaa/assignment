# assignment

### project structure
```
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models
│   └── logs           <- Training logs
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── src                <- Source code for use in this project.
│   │
│   ├── config         <- Models and data configurations files
│   │   
│   ├── data           <- Scripts to download, load or generate data│   │
│   │
│   ├── helpers        <- Files gathering useful functions together
│   │
│   ├── models         <- Files with models architectures
│   │
│   │ 
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── train.py       <- File used for running training procedure
│   │
│   ├── predict.py     <- File used for plotting example predictions or reconstructions
│   │
│   └── getResults.py  <- File used for printing confusion matrix, f1 and acc
│
├── README.md          <- The top-level README.│
├── requirements-conda 
└── requirements-pip.txt   <- The requirements file for reproducing the analysis environment, e.g.
                               generated with `pip freeze > requirements.txt`
```

### Installing development requirements
   
    conda install requirements-conda.txt
    pip install -r requirements-pip.txt

### How to run

    All models configuration should be set in configuration file located in src/config.
    To run training session use train.py with one of the following required arguments:
    - 'autoencoder' (to train only autoencoder)
    - 'classifier' (to train only classifier part of whole model which is connedted with encoder)
    - 'classifier_and_encoder' (to train whole model)


    To get confusion matrix, F1 score and Accuracy for test data use getResults.py with one of the following required arguments:
    - 'classifier' 
    - 'classifier_and_encoder' 

    To plot example predictions or reconstructions run predict.py with one of the following required arguments:
    - 'autoencoder'
    - 'classifier'
    - 'classifier_and_encoder'

    To run scripts manual downloading cifar-10 dataset is required. Script which would download the data has not been implemented yet. Data should be put into data/raw/