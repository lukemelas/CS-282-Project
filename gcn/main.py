import os, time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import load_cora
from train import train, validate
from model import GCN

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='use gpu if available')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--epochs', type=int, default=400, help='num training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--layers', type=int, default=2, help='# NN layers')
parser.add_argument('--h_size', type=int, default=16, help='# hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
parser.add_argument('--val_every', type=int, default=-1, help='val after X epochs')
parser.add_argument('--val_only', type=int, default=0, help='evaluate only') 
parser.add_argument('--checkpoint', type=str, default=None, help='save model to path')
parser.add_argument('--model', type=str, default=None, help='load saved model')
parser.add_argument('--log_every', type=int, default=50, help='print after X epochs') 
parser.add_argument('--prepro', type=str, default='../data/cora/preprocessed.pth') 
parser.add_argument('--test', type=int, default=0, help='use test data')

# Setup
start = time.time()
args = parser.parse_args()
args.gpu = args.gpu and torch.cuda.is_available()
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

# Data
if args.prepro:
    tmp = torch.load(args.prepro)
    adj_i, adj_v, adj_s, feats, labels, idx_train, idx_val, idx_test = tmp
    adj = torch.sparse.FloatTensor(adj_i, adj_v, adj_s)
else:
    adj, feats, labels, idx_train, idx_val, idx_test = load_cora()

# Model
model = GCN(num_layers=args.layers, 
            in_size=feats.shape[1],
            h_size=args.h_size, 
            out_size=labels.max().item() + 1,
            dropout=args.dropout)
if args.model:
    model.load_state_dict(torch.load(args.model))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

# GPU
if args.gpu:
    tmp = model, adj, feats, labels, idx_train, idx_val, idx_test
    tmp = [x.cuda() for x in tmp]
    model, adj, feats, labels, idx_train, idx_val, idx_test = tmp

# Train/Validate
print('Loaded data in {:.2f}s'.format(time.time() - start))
if args.test:
    assert args.model is not None, 'No model to evaluate'
    loss, acc = validate(model, adj, feats, labels, idx_test)
    print('Test complete. Loss: {:.4f} \tAcc: {:.4f}'.format(loss, acc))
elif args.val_only:
    assert args.model is not None, 'No model to evaluate'
    loss, acc = validate(model, adj, feats, labels, idx_val)
    print('Val complete. Loss: {:.4f} \tAcc: {:.4f}'.format(loss, acc))
else:
    loss, acc = train(model, optimizer, adj, feats, labels, idx_train, idx_val, args)
    print('Training complete. Best val loss: {:.4f}, acc: {:.4f}'.format(loss, acc))

