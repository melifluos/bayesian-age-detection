"""
Features generation code. This requires two files. One mapping ID -> label and an edge list with columns (input_vertex,
output_vertex)
"""
import utils
import pandas as pd
import numpy as np


def preprocess_income_data(label_path, edge_path, x_path, y_path, edge_list_path):
    targets = pd.read_csv(label_path, sep=' ')
    targets.columns = ['out_id', 'mean_income']
    print 'target labels of shape: ', targets.shape
    edges = pd.read_csv(edge_path)
    edges.columns = ['out_id', 'in_id']
    print 'edge list of shape: ', edges.shape
    all_data = edges.merge(targets)
    print 'all data of shape: ', all_data.shape
    X, y, edge_list = preprocess_data(all_data)
    utils.persist_edgelist(edge_list, edge_list_path)
    utils.persist_data(x_path, y_path, X, y)


def preprocess_data(input_data):
    """
    Reads a csv with columns out_id in_id in_idx num_followers cat weight
    Removes duplicates and creates and produces data in standard machine learning format X,y
    :param path: path to the training data
    :return: sparse csc matrix X of [out_idx,in_idx]
    :return: numpy array y of target categories
    """
    input_data = input_data.drop_duplicates(['out_id', 'in_id'])
    print 'input data of shape: ', input_data.shape, ' after duplicate removal'
    # replace the out ids with an index
    out_ids = input_data['out_id'].drop_duplicates()
    out_idx = np.arange(len(out_ids))
    out_lookup = pd.DataFrame(data={'out_id': out_ids.values, 'out_idx': out_idx}, index=out_idx)
    with_out_idx = input_data.merge(out_lookup, 'left')
    print 'input data of shape: ', with_out_idx.shape, ' after adding out idx'
    # add in index
    in_ids = input_data['in_id'].drop_duplicates()
    in_idx = np.arange(len(in_ids))
    in_lookup = pd.DataFrame(data={'in_id': in_ids.values, 'in_idx': in_idx}, index=in_idx)
    all_data = with_out_idx.merge(in_lookup, 'left')
    print 'input data of shape: ', all_data.shape, ' after adding in idx'
    edge_list = all_data[['out_idx', 'in_idx']]
    edge_list.columns = ['out_idx', 'in_idx']
    all_data.set_index('out_id', inplace=True)
    y = all_data[['out_idx', 'mean_income']].drop_duplicates()
    y = relabel(y)
    X = utils.edge_list_to_sparse_mat(edge_list)
    return X, y, edge_list


def relabel(y):
    uniq = pd.unique(y['mean_income'])
    uniq.sort()
    if uniq.max() > uniq.shape[0]:
        print('Relabeling y')
        bins = np.array(range(1, uniq.shape[0] + 1))
        income_bins = [bins[uniq == i[1]][0] for i in y['mean_income'].iteritems()]
        return pd.DataFrame(data={'out_idx': y['out_idx'], 'mean_income': income_bins})
    else:
        return y


if __name__ == '__main__':
    # input paths
    label_path = '../../resources/users-income'
    edge_path = '../../resources/users_friends.csv'
    # output paths
    edge_list_path = '../../resources/income.edgelist'
    x_path = '../../resources/income_X.p'
    y_path = '../../resources/income_y.p'
    preprocess_income_data(label_path, edge_path, x_path, y_path, edge_list_path)
