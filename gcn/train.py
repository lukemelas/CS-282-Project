import os, time
import torch 
import torch.nn as nn
import torch.nn.functional as F

def train(model, optimizer, adj, feats, labels, idx_train, idx_val, args):
    '''Train model, validating after every epoch'''
    start = time.time()
    best_loss, best_acc = 1e5, -1
    for epoch in range(args.epochs):

        # Train
        train_loss, train_acc, elapsed = train_epoch(model, optimizer, adj, feats, 
                                                  labels, idx_train, args, epoch)
        if epoch % args.log_every == 0:
            print('Epoch {:5d} \t\t| \tTrain loss: {:.4f} \tTrain acc: {:.4f}\tTime: {:.2f}'.format(
                   epoch+1, train_loss.item(), train_acc.item(), elapsed))

        # Validate
        if ((args.val_every > 0 and epoch % args.val_every == 0) or 
            (args.val_every < 0 and epoch == args.epochs - 1)):

            val_loss, val_acc = validate(model, adj, feats, labels, idx_val, args, epoch)
            print('Val acc: \t{:.4f}\t| \tValid loss: {:.4f}'.format(
                   val_acc, val_loss))
            if val_acc > best_acc:
               best_acc, best_loss = val_acc, val_loss

    return best_loss, best_acc

def train_epoch(model, optimizer, adj, feats, labels, idx_train, args, epoch):
    '''A single epoch of training (no minibatchingi)'''
    model.train()

    # Forward
    start = time.time()
    optimizer.zero_grad()
    output = model(feats, adj)

    # Backward
    loss = F.cross_entropy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    # Stats
    acc = accuracy(output[idx_train], labels[idx_train])
    time_elapsed = time.time() - start
    return loss, acc, time_elapsed
    
def validate(model, adj, feats, labels, idx_val, args, epoch):
    '''Validate on entire validation set (no minibatching)'''
    model.eval()
    output = model(feats, adj)
    loss = F.cross_entropy(output[idx_val], labels[idx_val])
    acc = accuracy(output[idx_val], labels[idx_val])
    return loss, acc

def accuracy(y_hat, y):
    '''Accuracy of prediction (max of prediction)'''
    preds = y_hat.max(1)[1].type_as(y)
    acc = preds.eq(y).float().sum() / len(y)
    return acc


