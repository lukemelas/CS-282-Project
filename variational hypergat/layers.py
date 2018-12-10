import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as ds
import pdb


class VariationalAttentionNet(nn.Module):
    def __init__(self, nfeat=128, nhid=16, nout=1, dropout=0.5):
        super(VariationalAttentionNet, self).__init__()
        self.lin1 = nn.Linear(nfeat * 2, nhid)
        self.lin21 = nn.Linear(nhid, nout)
        self.lin22 = nn.Linear(nhid, nout)
        self.dropout = dropout

    def get_log_p_minus_log_q(self, q, p, z):
        return p.log_prob(z).sum() - q.log_prob(z).sum()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.lin1(x))

        # Create attention distribution
        mu = self.lin21(x)
        # print("mu", mu)
        lv = self.lin22(x)
        # print("var", F.softplus(lv))
        # pdb.set_trace()

        # Sample z
        q = ds.Normal(mu, F.softplus(lv))
        p = ds.Normal(torch.zeros(1).cuda(), torch.ones(1).cuda() )
        if self.training:
            z = q.rsample() + 0 * torch.randn(mu.shape).cuda() # q.rsample()
            # print("z", z)
        else:
            z = q.loc
            # print("z", z)
        qmp = 0 - self.get_log_p_minus_log_q(q, p, z) 
        return z, qmp

class AttentionNet(nn.Module):
    def __init__(self, nfeat=128, nhid=16, nout=1, dropout=0.5):
        super(AttentionNet, self).__init__()
        self.lin1 = nn.Linear(nfeat * 2, nhid)
        self.lin2 = nn.Linear(nhid, nout)
        self.dropout = dropout
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, 0

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, attn='variational'):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        print((in_features, out_features))
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
       
        if attn == 'variational': 
            self.att = VariationalAttentionNet(out_features)
        else:
            self.att = AttentionNet(out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # e = self.leakyrelu(torch.matmul(h, h.t()))
        e, pmq = self.att(h, h)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime), pmq
        else:
            return h_prime, pmq

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)).cuda())
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
