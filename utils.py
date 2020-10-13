import torch
import os
import shutil
from typing import Tuple, List

from torch.utils.data import DataLoader

from config import EPSILON, PATH
from core import create_nshot_task_label, NShotTaskSampler


def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass


def setup_dirs():
    """Creates directories for this project."""
    mkdir(PATH + '/logs/')
    mkdir(PATH + '/logs/proto_nets')
    mkdir(PATH + '/logs/matching_nets')
    mkdir(PATH + '/logs/maml')
    mkdir(PATH + '/models/')
    mkdir(PATH + '/models/proto_nets')
    mkdir(PATH + '/models/matching_nets')
    mkdir(PATH + '/models/maml')


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def copy_weights(from_model: torch.nn.Module, to_model: torch.nn.Module):
    """Copies the weights from one model to another model.

    # Arguments:
        from_model: Model from which to source weights
        to_model: Model which will receive weights
    """
    if not from_model.__class__ == to_model.__class__:
        raise(ValueError("Models don't have the same architecture!"))

    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()


def autograd_graph(tensor: torch.Tensor) -> Tuple[
            List[torch.autograd.Function],
            List[Tuple[torch.autograd.Function, torch.autograd.Function]]
        ]:
    """Recursively retrieves the autograd graph for a particular tensor.

    # Arguments
        tensor: The Tensor to retrieve the autograd graph for

    # Returns
        nodes: List of torch.autograd.Functions that are the nodes of the autograd graph
        edges: List of (Function, Function) tuples that are the edges between the nodes of the autograd graph
    """
    nodes, edges = list(), list()

    def _add_nodes(tensor):
        if tensor not in nodes:
            nodes.append(tensor)

            if hasattr(tensor, 'next_functions'):
                for f in tensor.next_functions:
                    if f[0] is not None:
                        edges.append((f[0], tensor))
                        _add_nodes(f[0])

            if hasattr(tensor, 'saved_tensors'):
                for t in tensor.saved_tensors:
                    edges.append((t, tensor))
                    _add_nodes(t)

    _add_nodes(tensor.grad_fn)

    return nodes, edges


def prepare_meta_batch(n, k, q, meta_batch_size, stage, device, num_input_channels=3):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        if stage == 2:
            # evaluate the fast model on and generate meta-gradients
            x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
            
            # make k-way targets for stage two
            y = create_nshot_task_label(k, q).repeat(meta_batch_size)
        
        # Move to device
        x = x.double().to(device)
        y = y.to(device)
        return x, y

    return prepare_meta_batch_


def make_task_loader(dataset, args, train, meta):
    if meta:
        batch_size, n, k, q = args.meta_batch_size, args.n, args.k, args.q
    else:
        batch_size, n, k, q = args.batch_size, 1, 1, 0
    
    num_batches = args.epoch_len if train else args.eval_batches
    sampler = NShotTaskSampler(dataset, num_batches, n=n, k=k, q=q, num_tasks=batch_size)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers)


def correlation(x, y):
    """
    x has shape (num_samples, x_features)
    y has shape (num_samples, y_features)
    returns a matrix of correlatio coefficients of shape (x_features, y_features)
    """
    
    zx = (x - x.mean(dim=0, keepdim=True)) / x.var(dim=0, keepdim=True).sqrt()
    zy = (y - y.mean(dim=0, keepdim=True)) / y.var(dim=0, keepdim=True).sqrt()
    r = (zx.t() @ zy) / (x.shape[0] - 1)
    r[torch.isnan(r)] = 0.
    return r


def adjust_learning_rate(optimizer, epoch, lr):
    """copied from https://github.com/pytorch/examples"""
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr *= 0.1 ** (epoch // 200)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

