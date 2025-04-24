import utils
import torch
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp




def get_dataset(dataset):




    file_path = './dataset/' + dataset + '.pt'

    data_list = torch.load(file_path)

    # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
    adj = data_list[0]
    features = data_list[1]

    labels = data_list[2]

    idx_train = data_list[3]  
    idx_val = data_list[4]
    idx_test = data_list[5]


    adj = utils.torch_sparse_tensor_to_sparse_mx(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = utils.normalize_adj(adj)
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)



    return adj, features, labels, idx_train, idx_val, idx_test



