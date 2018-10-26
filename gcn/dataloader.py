import os
import numpy as np
import torch
import scipy.sparse as sp_sparse
import pdb

from utils import scipy_to_torch_sparse, normalize_sparse, \
                  labels_to_onehot, coo_to_symmetric


def load_cora(path='../data/cora/'):
    '''Load CORA: https://relational.fit.cvut.cz/dataset/CORA
       In the content file, the 1st column is the paper id, the 2nd
       to (n-1)th are the features, and the nth is the category/label.
       The cites file gives the edges in the graphs of papers'''

    # Load data
    data = np.genfromtxt(path+'cora.content', dtype=np.dtype(str))
    feats = sp_sparse.csr_matrix(data[:,1:-1], dtype=np.float32)
    labels = labels_to_onehot(data[:,-1])
    N, D = len(labels), feats.shape[0] # num nodes, num features

    # Build graph
    idx_map = {j: i for i,j in enumerate(np.array(data[:,0], dtype=np.int32))}
    edges = np.genfromtxt(path+'cora.cites', dtype=np.int32) # load edges
    edges = np.array(list(map(idx_map.get, edges.flatten())), 
                     dtype=np.int32).reshape(edges.shape) # order edges
    adj = (np.ones(edges.shape[0]), (edges[:,0], edges[:,1])) # all edges weight 1
    adj = sp_sparse.coo_matrix(adj, shape=(N, N), dtype=np.float32) # --> sparse matrix
    adj = coo_to_symmetric(adj) # --> symmetric adjacency matrix --> wait why?? 
    adj = adj + sp_sparse.eye(adj.shape[0]) # add identity (self-links)

    # Preprocessing
    feats = normalize_sparse(feats)
    adj = normalize_sparse(adj)

    # Train/val/test splits
    idx_train = torch.LongTensor(range(140))
    idx_val   = torch.LongTensor(range(200,500))
    idx_test  = torch.LongTensor(range(500,1500))

    # Torchify
    feats = torch.from_numpy(np.array(feats.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = scipy_to_torch_sparse(adj)

    return adj, feats, labels, idx_train, idx_val

# For testing
if __name__ == '__main__':
    import time
    start = time.time()
    if False:
        adj, feats, labels, idx_train, idx_val = load_cora()
        adj_i, adj_v, adj_s = adj._indices(), adj._values(), adj.shape
        data = (adj_i, adj_v, adj_s, feats, labels, idx_train, idx_val)
        torch.save(data, '../data/cora/preprocessed.pth')
        print('Saved cora in {:.2f}s'.format(time.time() - start))
    else:
        load_cora()
        print('Loaded cora in {:.2f}s'.format(time.time() - start))
        
