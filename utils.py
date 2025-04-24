import torch
import numpy as np
import scipy.sparse as sp
from torch.nn.functional import normalize
import dgl
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def normwithslef(adj):

    adj = torch_sparse_tensor_to_sparse_mx(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def auc(output, labels):
    auc_score = roc_auc_score(y_true=labels.cpu().detach().numpy(),
            y_score=output.cpu().detach().numpy())   

    return auc_score 

def auc_batch(output, labels):
    auc_score = roc_auc_score(y_true=labels.cpu().detach().numpy(),
            y_score=output.cpu().detach().numpy())   

    return auc_score * labels.shape[0]

from sklearn.metrics import roc_auc_score, accuracy_score
def roc_auc_score_FIXED(output, labels):
    y_true=labels.cpu().detach().numpy()
    y_score=output.cpu().detach().numpy()
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_score)) * labels.shape[0]
    return roc_auc_score(y_true, y_score) * labels.shape[0]



def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    return lap_pos_enc



def re_features(adj, features, K):
    #传播之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    for i in range(features.shape[0]):

        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)

    for i in range(K):

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]

    nodes_features = nodes_features.squeeze()

    return nodes_features






def node_neighborhood_feature(adj, features, k):

    x_0 = features
    for i in range(k):

        features = 0.9 * torch.mm(adj, features) + 0.1 * x_0

    return features



def node_seq_feature(features, pk, nk, sample_batch):


    #采样之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features_p = torch.empty(features.shape[0], pk+1, features.shape[1])

    nodes_features_n = torch.empty(features.shape[0], nk+1, features.shape[1])

    x = features + torch.zeros_like(features)
    
    x = normalize(x, dim=1)


    #构建batch采样
    total_batch = int(features.shape[0]/sample_batch)

    rest_batch = int(features.shape[0]%sample_batch)

    for index_batch in tqdm(range(total_batch)):

        # x_batch, [b,d]
        # x1_batch, [b,d]
        # 切片操作左闭右开
        x_batch = x[(index_batch)*sample_batch:(index_batch+1)*sample_batch,:]

        
        s = torch.matmul(x_batch, x.transpose(1, 0))


        # print("------------begin sampling positive samples------------")

        #采正样本
        print(s.shape)
        for i in range(sample_batch):
            s[i][(index_batch)*(sample_batch) + i] = -1000        #将得到的相似度矩阵的对角线的值置为负的最大值

        topk_values, topk_indices = torch.topk(s, pk, dim=1)
        
        # print(topk_indices)


        for index in range(sample_batch):

            # print((index_batch)*sample_batch + index)

            nodes_features_p[(index_batch)*sample_batch + index, 0, :] = features[(index_batch)*sample_batch + index]
            for i in range(pk):
                nodes_features_p[(index_batch)*sample_batch + index, i+1, :] = features[topk_indices[index][i]]


        # print("------------begin sampling negative samples------------")

        #采负样本

        if nk > 0:
            all_idx = [i for i in range(s.shape[1])]

            for index in tqdm(range(sample_batch)):

                nce_idx = list(set(all_idx) - set(topk_indices[index].tolist()))
                

                nce_indices = np.random.choice(nce_idx, nk, replace=True)

                # print((index_batch)*sample_batch + index)
                # print(nce_indices)


                nodes_features_n[(index_batch)*sample_batch + index, 0, :] = features[(index_batch)*sample_batch + index]
                for i in range(nk):
                    nodes_features_n[(index_batch)*sample_batch + index, i+1, :] = features[nce_indices[i]]



    if rest_batch > 0:

        x_batch = x[(total_batch)*sample_batch:(total_batch)*sample_batch + rest_batch,:]


        s = torch.matmul(x_batch, x.transpose(1, 0))

        print("------------begin sampling positive samples------------")

        #采正样本
        for i in range(rest_batch):
            s[i][(total_batch)*sample_batch + i] = -1000         #将得到的相似度矩阵的对角线的值置为负的最大值

        topk_values, topk_indices = torch.topk(s, pk, dim=1)
        # print(topk_indices.shape)

        for index in range(rest_batch):
            nodes_features_p[(total_batch)*sample_batch + index, 0, :] = features[(total_batch)*sample_batch + index]
            for i in range(pk):
                nodes_features_p[(total_batch)*sample_batch + index, i+1, :] = features[topk_indices[index][i]]


        print("------------begin sampling negative samples------------")

        #采负样本
        if nk > 0:
            all_idx = [i for i in range(s.shape[1])]

            for index in tqdm(range(rest_batch)):

                nce_idx = list(set(all_idx) - set(topk_indices[index].tolist()))
                

                nce_indices = np.random.choice(nce_idx, nk, replace=False)
                # print(nce_indices)

                # print((index_batch)*sample_batch + index)
                # print(nce_indices)


                nodes_features_n[(total_batch)*sample_batch + index, 0, :] = features[(total_batch)*sample_batch + index]
                for i in range(nk):
                    nodes_features_n[(total_batch)*sample_batch + index, i+1, :] = features[nce_indices[i]]


    nodes_features = torch.concat((nodes_features_p, nodes_features_n), dim=1)
    

    print(nodes_features_p.shape)
    print(nodes_features_n.shape)
    print(nodes_features.shape)

    return nodes_features






























 








