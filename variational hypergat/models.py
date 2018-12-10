import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer

import pdb

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        attention_outs = [att(x, adj) for att in self.attentions]
        zs = [t[0] for t in attention_outs]
        pmq = sum(t[1] for t in attention_outs)

        x = torch.cat(zs, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x, pmq2 = self.out_att(x, adj)
        pmq = pmq + pmq2
        x = F.elu(x)
        return F.log_softmax(x, dim=1), pmq


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions): self.add_module('attention_{}'.format(i), attention)

        self.attentions2 = [SpGraphAttentionLayer(nfeat * nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] # ADD
        for i, attention in enumerate(self.attentions): self.add_module('attention2_{}'.format(i), attention) # ADD

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1) # ADD
        x = F.dropout(x, self.dropout, training=self.training) # ADD
        x = self.out_att(x, adj) # F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

