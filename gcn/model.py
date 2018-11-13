import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    '''Graph convolution --> essentially a sparse linear layer'''
    def __init__(self, in_feats, out_feats, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.zeros(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_feats))
        else:
            self.register_parameters('bias', None)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None: 
            self.bias.data.zero_()

    def forward(self, input, adj):
        sup = input @ self.weight   # support
        out = torch.spmm(adj, sup)  # combine with neighbors
        b = self.bias if self.bias is not None else 0
        return out + b

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_feats)    + ' -> ' \
               + str(self.out_feats)   + ')'

class GCN(nn.Module):
    '''Graph convolution network'''
    def __init__(self, num_layers, in_size, h_size, out_size, dropout=0.3,
                 mc_dropout=False):
        super(GCN, self).__init__()
        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.dropout = dropout
        self.mc_dropout = mc_dropout
        
        # Create graph convolutional layers
        self.gcns = nn.ModuleList()
        for i in range(num_layers):
            in_feats = in_size if i == 0 else h_size
            out_feats = out_size if i == num_layers - 1 else h_size
            self.gcns.extend([GraphConvolution(in_feats, out_feats)])

    def forward(self, x, adj):
        '''ReLU nonlinearity and dropout'''
        use_dropout = (not self.training) if self.mc_dropout else self.training
        for i in range(len(self.gcns) - 1):
            x = self.gcns[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=use_dropout)
        x = self.gcns[-1](x, adj)
        return x

