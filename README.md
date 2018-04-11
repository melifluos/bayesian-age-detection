# bayesian-age-detection

This is code acompanying the paper 'Probabilistic Inference of Twitter Users' Age based on What They Follow' https://pdfs.semanticscholar.org/8db1/5d1ab276fd5460b6ecccb5354655aa6ee7bd.pdf. These instructions describe how to run the bayesian model to detect the age of Twitter users. This model uses a hierarchical Bayesian framework to generalise from several thousand labelled examples to predict the age of 700 million Twitter users. Labelled data was mined from Twitter description fields using the included regex. For testing we include a sample of 30k anonymised labelled accounts with this repo.

## Getting Started

To run the model clone the repo 

cd to the project's root folder

python src/python/age_detector.py resources/features.p resources/labels.p -nfolds 3

### Prerequisites

The code uses the numpy, pandas and scikit-learn python packages. We recommend installing these through Anaconda

### Data

For privacy reasons we can't include the raw data age data. Instead we include two files

(1) features.p is a pandas pickle file of a scipy sparse matrix of shape (31389, 50190). Each row is a single Twitter user with a labelled age. Each column is a Twitter user followed by more than 10 of the labeled accounts

(2) labels.p is a pandas pickle file of a pandas dataframe of shape(31389, 2). Each row is an index into a row of features.p and an age label in {1,2,3,4,5,6,7} corresponding to an age of {10-20,20-30,30-40,40-50,50-60,60-70,70-80}. The data is sampled from a larger dataset in such a way that there are roughly equal numbers of each age class.


To read the data:

import pandas

x = pd.read_pickle('features.p')

y = pd.read_pickle('labels.p')

To increase the general utility of the code, we also include our pre-processing script and a public sample of Twitter data with labelled incomes. To generate features for this data set navigate to the src/python folder and type 

python generate_features

into a terminal. This will create income_X.p and income_y.p files in the resources folder, which have the same format as the Twitter age data.

## Authors

**Ben Chamberlain**

## Citation

If this code helps, please cite:

Chamberlain, Benjamin Paul, Clive Humby, and Marc Peter Deisenroth. "Probabilistic Inference of Twitter Users’ Age based on What They Follow." Joint European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases. Springer International Publishing, 2017.


