import os
import numpy as np
import torch
import scipy.sparse as sp_sparse
import pdb
import scipy.sparse as sp


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
    adj_orig = adj # for transfering model to gae
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

    ## Save for GAE
    #torch.save({'adj_orig': adj_orig.todense(), 'adj_i': adj.indices, 'adj_v': adj.values,
    #            'adj_s': adj.shape, 'feats': feats, 'labels': labels, 
    #            'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}, 
    #            '../data/cora/preprocessed_gcn_data_for_gae.pth')

    return adj, feats, labels, idx_train, idx_val, idx_test

def load_adv_data(node, path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp_sparse.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp_sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    labels = np.where(encode_onehot(idx_features_labels[:, -1]))[1]
    new_adj = deepcopy(np.array(adj.todense()))
    neighbours = (new_adj[node, :] > 0).astype(int)
    neighbours[node] = 0
    not_class = (labels != labels[node]).astype(int)
    eligible = np.where(not_class - neighbours > 0)[0] #indices to choose new edges from
    ineligible = np.where(neighbours)[0] #indices to remove edges from
    
    for i in ineligible:
        if i in eligible:
            import pdb; pdb.set_trace()
        assert(i not in eligible)

    num_perturb = min(int(sum(neighbours) / 2 + 1), len(eligible), len(ineligible))
    to_add = eligible[np.random.choice(len(eligible), num_perturb, replace=False)]
    to_remove = ineligible[np.random.choice(len(ineligible), num_perturb, replace=False)]
    print("Added: ", to_add, "Removed: ", to_remove)

    for idx in to_add:
        assert(new_adj[node,idx] == 0)
        assert(new_adj[idx,node] == 0)
        new_adj[node,idx] = 1
        new_adj[idx,node] = 1

    for idx in to_remove:
        assert(new_adj[node,idx] == 1)
        assert(new_adj[idx,node] == 1)
        new_adj[node,idx] = 0
        new_adj[idx,node] = 0
    
    adj = sp_sparse.csr_matrix(new_adj)
    features = normalize(features)
    adj = normalize(adj + sp_sparse.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# For testing
if __name__ == '__main__':
    import time
    start = time.time()
    if False:
        adj, feats, labels, idx_train, idx_val, idx_test = load_cora()
        adj_i, adj_v, adj_s = adj._indices(), adj._values(), adj.shape
        data = (adj_i, adj_v, adj_s, feats, labels, idx_train, idx_val, idx_test)
        torch.save(data, '../data/cora/preprocessed.pth')
        print('Saved cora in {:.2f}s'.format(time.time() - start))
    else:
        load_cora()
        print('Loaded cora in {:.2f}s'.format(time.time() - start))
        
