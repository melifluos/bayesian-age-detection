import pandas as pd
import cPickle as pickle
import numpy as np
from sklearn.metrics import f1_score
from scipy.sparse import lil_matrix


__author__ = 'benchamberlain'


def read_data(x_path, y_path, threshold):
    """
    reads the features and target variables
    :return:
    """
    targets = read_pickle(y_path)
    try:
        y = np.array(targets['cat'])
    except KeyError:  # doing income instead of age
        y = np.array(targets['mean_income'])
    X = read_pickle(x_path)
    X1, cols = remove_sparse_features(X, threshold=threshold)
    print X1.shape
    return X1, y


def read_pickle(path):
    with open(path, 'rb') as infile:
        return pd.read_pickle(infile)


def edge_list_to_sparse_mat(edge_list):
    """
    Convert a pandas DF undirected edge list for a bipartite graph into a scipy csc sparse matrix.
    Assumes that edges are contiguosly indexed starting at 0
    :param edge_list: A pandas DF with columns [fan_idx, star_idx]
    :return: A Columnar sparse matrix
    """
    # Create matrix representation (adjacency matrix) of edge list
    data_shape = edge_list.max(axis=0)
    print 'building sparse matrix of size {0}'.format(data_shape)
    X = lil_matrix((data_shape['fan_idx'] + 1, data_shape['star_idx'] + 1), dtype=int)
    X[edge_list['fan_idx'].values, edge_list['star_idx'].values] = 1
    return X.tocsc()


def remove_sparse_features(sparse_mat, threshold):
    """
    removes features (stars) with less than threshold observations in this data set
    :param X:
    :param threshold:
    :return: A version of X with columns that are too sparse removed and a list of the good column indices
    """
    print 'input matrix of shape: {0}'.format(sparse_mat.shape)
    observations = np.array(sparse_mat.sum(axis=0)).flatten()
    good_cols = np.where(observations >= threshold)[0]
    out_mat = sparse_mat[:, good_cols]
    print 'output matrix of shape: {0}'.format(out_mat.shape)
    return out_mat, good_cols


def get_metrics(y, pred, verbose=False):
    """
    generate metrics to assess the detectors
    :param y:
    :param pred:
    :return:
    """
    macro_f1 = f1_score(y, pred, average='macro')
    micro_f1 = f1_score(y, pred, average='micro')
    all_scores = f1_score(y, pred, average=None)
    if verbose:
        print 'macro'
        print macro_f1
        print 'micro'
        print micro_f1
        scores = np.zeros(shape=(1, len(all_scores)))
        scores[0, :] = all_scores
        print pd.DataFrame(data=scores, index=None, columns=np.arange(len(all_scores)))
    return macro_f1, micro_f1
