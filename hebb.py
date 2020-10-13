import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union

from core import create_nshot_task_label, EvaluateFewShot
from metrics import categorical_accuracy
from utils import correlation


def least_dist(x, y, v=0.):
    # x has shape (meta_batch_size, mesa_batch_size, num_features)
    # y has shape (meta_batch_size, mesa_batch_size, num_classes)
    # v has shape (meta_batch_size, num_classes, num_features)
    xt = x.transpose(1, 2)
    xxtinv = torch.inverse(torch.bmm(x, xt))
    xxtinvx = torch.bmm(xxtinv, x)
    yt = y.transpose(1, 2)
    
    if type(v) is torch.Tensor:
        vxt = torch.bmm(v, xt)
        w = v + torch.bmm(yt - vxt, xxtinvx)
    else:
        w = torch.bmm(yt, xxtinvx)
    
    return w


def stage_one_least_norm(model: Module,
                         optimiser: Optimizer,
                         loss_fn: Callable,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         n_shot: int,
                         k_way: int,
                         q_queries: int,
                         inner_train_steps: int,
                         inner_lr: float,
                         hebb_lr: float,
                         train: bool,
                         device: Union[str, torch.device]):
    """
    supervised training on all 64 classes of the meta training set
    validation on a n-shot k-way meta-learning task from the meta-validation set 
    validation uses a least-norm learner as meta-learning algorithm
    for training
      x has shape (meta_batch_size * (n*k + q*k), channels, width, height)
      y is an int in {0, ..., 64} and has shape (meta_batch_size * (n*k + q*k))
    for validation
      x has shape (meta_batch_size, (n*k + q*k), channels, width, height)
      y should to be constructed locally
    """
    args = {'device': device, 'dtype': torch.double}
    model.train(train)
    
    if train:
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    else:
        """
        evaluate stage one model on meta validation set
        that is we construct a least norm learner on top of the pretrained features
        """
        
        with torch.no_grad():
            model(x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]))
            # features of shape (meta_batch_size, n*k + q*k, num_features)
            features = model.features.reshape(x.shape[0], x.shape[1], -1)
            # make support features z of shape (meta_batch_size, n*k, num_features+1)
            z = features[:,:n_shot*k_way]
            z = torch.cat([z, torch.ones_like(z[:,:,:1])], dim=2) # add bias unit
            # make labels of shape (meta_batch_size, n*k, k)
            y = create_nshot_task_label(k_way, n_shot).to(device)
            y = y.repeat(x.shape[0])
            y = (torch.eye(k_way, **args)[y,:] * 2 - 1) * 10
            y = y.reshape(x.shape[0], k_way*n_shot, -1)
            # least norm solutions of shape (meta_batch_size, k, num_features+1)
            zztinv = torch.inverse(torch.bmm(z, z.transpose(1, 2)))
            w = torch.bmm(y.transpose(1, 2), torch.bmm(zztinv, z))
            # query features z of shape (meta_batch_size, q*k, num_features+1)
            z = features[:,n_shot*k_way:]
            z = torch.cat([z, torch.ones_like(z[:,:,:1])], dim=2) # add bias unit
            # make labels of shape (meta_batch_size, q*k, k)
            y = create_nshot_task_label(k_way, q_queries).to(device)
            y = y.repeat(x.shape[0])
            y_target = y
            y = (torch.eye(k_way, **args)[y,:] * 2 - 1) * 10
            y = y.reshape(x.shape[0], k_way*q_queries, -1)
            y_hat = torch.bmm(z, w.transpose(1, 2))
            y_hat = y_hat.reshape(-1, k_way)
            loss = loss_fn(y_hat, y_target)
    
    return loss, y_hat


def __stage_two(model: Module,
              optimiser: Optimizer,
              loss_fn: Callable,
              x: torch.Tensor,
              y: torch.Tensor,
              n_shot: int,
              k_way: int,
              q_queries: int,
              inner_train_steps: int,
              inner_lr: float,
              hebb_lr: float,
              train: bool,
              device: Union[str, torch.device],
              sys2net = lambda x, y : (0., (0., 0.)), 
              sys2feat = lambda x : 1.):
    args = {'device': device, 'dtype': torch.double}
    meta_batch_size, mesa_batch_size = x.shape[0], x.shape[1]
    model.train(train)
    
    # actiate model.features
    model(x.reshape(meta_batch_size * mesa_batch_size, *x.shape[2:]))
    
    # features of shape (meta_batch_size, n*k + q*k, num_features + 1)
    features = model.features.reshape(meta_batch_size, mesa_batch_size, -1)
    features = torch.cat([features, torch.ones_like(features[:,:,:1])], dim=2)
    support_features = features[:,:n_shot*k_way]
    query_features = features[:,n_shot*k_way:]
    
    # make support labels of shape (meta_batch_size, n*k, k)
    y = create_nshot_task_label(k_way, n_shot).to(device)
    y = y.repeat(meta_batch_size)
    y = (torch.eye(k_way, **args)[y,:] * 2 - 1) * 10
    support_y = y.reshape(meta_batch_size, n_shot * k_way, -1)
    
    # make query labels of shape (meta_batch_size, q*k)
    y = create_nshot_task_label(k_way, q_queries).to(device)
    query_y = y.repeat(meta_batch_size)
    
    # get least distance solution on support set
    weight, bias = model.output_layer.weight, model.output_layer.bias
    v = torch.cat([weight, bias.reshape(-1, 1)], dim=1)
    v = v.unsqueeze(0).repeat(meta_batch_size, 1, 1)
    w = least_dist(support_features, support_y, v)
    
    # compute predictions and loss for query set
    query_y_hat = torch.bmm(query_features, w.transpose(1, 2))
    query_y_hat = query_y_hat.reshape(-1, k_way)
    loss = loss_fn(query_y_hat, query_y)
    predictions = query_y_hat.softmax(dim=1)
    
    if train:
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    return loss, predictions



def stage_two(model: Module,
              optimiser: Optimizer,
              loss_fn: Callable,
              x: torch.Tensor,
              y: torch.Tensor,
              n_shot: int,
              k_way: int,
              q_queries: int,
              inner_train_steps: int,
              inner_lr: float,
              hebb_lr: float,
              train: bool,
              device: Union[str, torch.device],
              sys2net = lambda x, y : (0., (0., 0.)), 
              sys2feat = lambda x : 1.):
    """
    meta-learning using hebb rule on output layer only
    """
    task_losses = []
    task_predictions = []
    model.train(train)
    
    for m in [sys2net, sys2feat]:
        if isinstance(m, nn.Module):
            m.train(train)
    
    # x has shape (meta_batch_size, n*k + q*k, channels, width, height)
    # TODO the loop below iterates over tasks. can we parallelize this?
    
    for meta_batch in x: 
        if True: # FIXME this is for miniImagenet, tieredImagenet, Omniglot
            x_task_train = meta_batch[:n_shot * k_way]
            x_task_val = meta_batch[n_shot * k_way:]
        else: # FIXME this is for xdom data
            meta_batch = meta_batch.reshape(k_way, n_shot+q_queries, *meta_batch.shape[1:])
            x_task_train = meta_batch[:,:n_shot]
            x_task_train = x_task_train.flatten(0,1)
            x_task_val = meta_batch[:,n_shot:]
            x_task_val = x_task_val.flatten(0,1)

        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())
        #outp_weights = (fast_weights['logits.weight'], fast_weights['logits.bias'])
        outp_weights = (fast_weights['output_layer.weight'], fast_weights['output_layer.bias'])
        
        y = create_nshot_task_label(k_way, n_shot).to(device)
        _ = model(x_task_train, output_layer=False) # update model.features
        
        if hasattr(sys2net, 'init_hidden'):
          (h, c) = sys2net.init_hidden(n_shot * k_way, device, torch.double)
        else:
          (h, c) = (0, 0)
        
        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(inner_train_steps):
            # compute labels and predictions for the support set
            if hasattr(model, 'dropout'):
                features = F.dropout(model.features, p=model.dropout.p, training=train)
            else:
                features = model.features
            
            # TODO use sys2net to update outp_weights, i.e. something like
            # outp_weights += sys2net(outp_weights)
            #print('feat', features.shape, 'weight', outp_weights[0].shape)
            logits = F.linear(features, *outp_weights)
            loss = loss_fn(logits, y)
            
            # adjust weights using Hebb rule
            d_logits = torch.autograd.grad(loss, (logits,), create_graph=train)[0]
            sys2out, (h, c) = sys2net(d_logits, (h, c))
            #d_logits = d_logits * (1 + sys2out)
            d_logits = d_logits + sys2out
            features = features * 2 * sys2feat(features)
            gradients = (d_logits.t() @ features, d_logits.sum(dim=0))
            print(gradients[0])
            
            hebb_lr = hebb_lr if inner_batch == 0 else inner_lr
            outp_weights = tuple(w - hebb_lr * g for w, g in zip(outp_weights, gradients))
            #fast_weights['logits.weight'], fast_weights['logits.bias'] = outp_weights
            fast_weights['output_layer.weight'], fast_weights['output_layer.bias'] = outp_weights
        
        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(k_way, q_queries).to(device)
        _ = model(x_task_val, output_layer=False)
        
        if hasattr(model, 'dropout'):
            features = F.dropout(model.features, p=model.dropout.p, training=train)
        else:
            features = model.features
        
        logits = F.linear(features, *outp_weights)
        #logits = model.functional_forward(x_task_val, fast_weights)
        loss = loss_fn(logits, y)
        
        #loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)

        # Accumulate losses and gradients
        task_losses.append(loss)
    
    meta_batch_loss = torch.stack(task_losses).mean()

    if train:
        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

    return meta_batch_loss, torch.cat(task_predictions)


def hebb_rule(model: Module,
              optimiser: Optimizer,
              loss_fn: Callable,
              x: torch.Tensor,
              y: torch.Tensor,
              n_shot: int,
              k_way: int,
              q_queries: int,
              inner_train_steps: int,
              hebb_lr: float,
              train: bool,
              device: Union[str, torch.device],
              xdom=False):
    # x has shape (meta_batch_size, n*k + q*k, channels, width, height)
    # TODO make a clean implementation of the simple hebb rule
    
    args = {'device': device, 'dtype': torch.double, 'requires_grad': True}
    task_predictions = []
    task_losses = []
    model.train(train)
    
    for x_ in x:
        # FIXME add slicing for xdom data
        if xdom:
          x_ = x_.reshape(k_way, n_shot+q_queries, *x_.shape[1:])
          x_support = x_[:,:n_shot].flatten(0,1)
          x_query = x_[:,n_shot:].flatten(0,1)
        else:
          x_support = x_[:n_shot * k_way]
          x_query = x_[n_shot * k_way:]
        
        y = create_nshot_task_label(k_way, n_shot).to(device)
        model(x_support, output_layer=False) # activate model.features
        features = model.module.features
        
        if not isinstance(features, list):
            features = [features]
        
        weights = []
        
        for f in features:
            weights.append((torch.zeros(k_way, f.shape[1], **args), 
                            torch.zeros(k_way, **args)))
            w = weights[-1]
            
            for i in range(inner_train_steps):
                logits = F.linear(f, *w)
                loss = loss_fn(logits, y)
                g = torch.autograd.grad(loss, w)
                
                # FIXME uncomment following line to reproduce old impl
                g = g[0] * 2, g[1]
                w = tuple(w_ - hebb_lr * g_ for w_, g_ in zip(w, g))
            
            weights[-1] = w
        
        y = create_nshot_task_label(k_way, q_queries).to(device)
        model(x_query, output_layer=False)
        features = model.module.features
        
        if not isinstance(features, list):
            features = [features]
        
        logits = [F.linear(f, *w) for f, w in zip(features, weights)]
        
        #print('-----------------------------------------------------------')
        #for l in logits:
        #    print(l.argmax(dim=1))
        
        
        #logits = [l.softmax(dim=1) for l in logits]
        logits = sum(logits) # ensemble by adding logits together
        
        #print(logits.argmax(dim=1))
        #print(y)
        #print('-----------------------------------------------------------')
        
        loss = loss_fn(logits, y)
        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)
        task_losses.append(loss)
    
    meta_batch_loss = torch.stack(task_losses).mean()
    return meta_batch_loss, torch.cat(task_predictions)















