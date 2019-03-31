import numpy as np
import torch
import scipy.sparse as sp_sparse
import pdb

def scipy_to_torch_sparse(x):
    '''Converts sparse scipy matrix to torch sparse tensor'''
    x = x.tocoo().astype(np.float32) # convert to COO format
    idx = np.vstack((x.row, x.col)).astype(np.int64)
    idx = torch.from_numpy(idx) # matrix indices
    val = torch.from_numpy(x.data) # matrix data
    t = torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))
    return t

def normalize_sparse(x):
    '''Fast sparse row-normalization'''
    row_sum_inv = np.power(np.array(x.sum(1)), -1).flatten()
    row_sum_inv[np.isinf(row_sum_inv)] = 0 # set 0s that went 1/0 back to 0
    row_sum_inv_matrix = sp_sparse.diags(row_sum_inv)
    x = row_sum_inv_matrix.dot(x)
    return x

def labels_to_onehot(labels):
    '''Labels list --> one hot matrix'''
    classes = set(labels)
    classes = {c: np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    onehots = np.array(list(map(classes.get, labels)), dtype=np.float32)
    return onehots

def coo_to_symmetric(x):
    '''Converts sparse (COO) matrix to symmetric adjacency matrix'''
    return x + x.T.multiply(x.T > x) - x.multiply(x.T > x)
       
def check(val, msg):
    '''Checks whether val is nan or inf and prints msg if True'''
    if not val: print(msg); pdb.set_trace()
        
def to_numpy(t):
    '''PyTorch tensor to numpy array'''
    return t.detach().to('cpu').numpy()

class AverageMeter(object):
    """Computes and stores the average, current value, 
       and a rolling average with window size roll_len"""
    def __init__(self, roll_len=100):
        self.roll_len = roll_len
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
        self.roll = []; self.roll_avg = 0

    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count
        self.roll = (self.roll + [val] * n)[-self.roll_len:]
        self.roll_avg = sum(self.roll) / len(self.roll)

