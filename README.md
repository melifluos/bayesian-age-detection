# bayesian-age-detection

These instructions describe how to run the bayesian model to detect the age of Twitter users. This model uses a hierarchical Bayesian framework to generalise from several thousand labelled examples to predict the age of 700 million Twitter users. Labelled data was mined from Twitter description fields using the included regex. For testing we include a sample of 30k anonymised labelled accounts with this repo.

## Getting Started

To run the model clone the repo 

cd to the project's root folder

python src/python/age_detector.py resources/features.p resources/labels.p -nfolds 3

### Prerequisites

The code uses the numpy, pandas and scikit-learn python packages. We recommend installing these through Anaconda

## Authors

**Ben Chamberlain**


