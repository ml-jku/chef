from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from typing import Dict
from collections import OrderedDict
import math


##########
# Layers #
##########
class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    """Performs global max pooling over the entire length of a batched 1D tensor

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    """Performs global average pooling over the entire height and width of a batched 2D tensor

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


def conv_block(in_channels: int, out_channels: int, no_relu: bool = False) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    
    seq = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ]
    
    if no_relu:
        del seq[2]
    
    return nn.Sequential(*seq)


def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor,
                          bn_weights, bn_biases, no_relu=False) -> torch.Tensor:
    """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.

    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    
    if not no_relu:
        x = F.relu(x)
    
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


##########
# Models #
##########
def get_few_shot_encoder(num_input_channels=1, conv128=False) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks

    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    
    mult = conv128 + 1
    
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64 * mult),
        conv_block(64 * mult, 64 * mult),
        Flatten(),
    )


def conv64():
    return get_few_shot_encoder(3, False)

class FewShotClassifier(nn.Module):
    def __init__(self, num_input_channels: int, k_way: int, final_layer_size: int = 64, 
                 dropout = 0., mult = 1, no_relu = False):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(FewShotClassifier, self).__init__()
        self.no_relu = no_relu
        self.final_layer_size = final_layer_size
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64 * mult)
        self.conv4 = conv_block(64 * mult, 64 * mult, self.no_relu)
        
        self.dropout = nn.Dropout(dropout)

        self.logits = nn.Linear(final_layer_size * mult, k_way)

    def forward(self, x, output_layer=True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        self.features = x
        
        if output_layer:
          x = self.dropout(x)

          return self.logits(x)
        else:
          return None

    def functional_forward(self, x, weights):
        """Applies the same forward pass using PyTorch functional operators using a specified set of weights."""

        for block in [1, 2, 3, 4]:
            x = functional_conv_block(
                x, 
                weights[f'conv{block}.0.weight'], 
                weights[f'conv{block}.0.bias'],
                weights.get(f'conv{block}.1.weight'), 
                weights.get(f'conv{block}.1.bias'),
                no_relu=False if block < 4 else self.no_relu
            )

        x = x.view(x.size(0), -1)
        self.features = x
        x = F.dropout(x, p=self.dropout.p, training=self.training)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x


def conv(k_way, dropout, no_relu):
    return FewShotClassifier(3, k_way, 1600, dropout=dropout, mult=1, no_relu=no_relu)


class MatchingNetwork(nn.Module):
    def __init__(self, n: int, k: int, q: int, fce: bool, num_input_channels: int,
                 lstm_layers: int, lstm_input_size: int, unrolling_steps: int, device: torch.device):
        """Creates a Matching Network as described in Vinyals et al.

        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels)
        if self.fce:
            self.g = BidrectionalLSTM(lstm_input_size, lstm_layers).to(device, dtype=torch.double)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=unrolling_steps).to(device, dtype=torch.double)

    def forward(self, inputs):
        pass


class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().double()
        c = torch.zeros(batch_size, embedding_dim).cuda().double()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h


class Sys2Net1(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Sys2Net1, self).__init__()
    self.lin1 = nn.Linear(input_size, hidden_size)
    self.gate = nn.Linear(input_size, hidden_size)
    self.lin2 = nn.Linear(hidden_size, output_size)
    nn.init.constant_(self.gate.bias, -3.)
    nn.init.constant_(self.lin2.bias,  0.)


  def forward(self, x):
    x = torch.tanh(self.lin1(x)) * torch.sigmoid(self.gate(x))
    return self.lin2(x)


class Sys2Net2(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_hidden_layer, relu_out, init0, dropout):
    super(Sys2Net2, self).__init__()
    
    self.relu_out = relu_out
    self.num_hidden_layer = num_hidden_layer
    
    self.lin = nn.ModuleList([nn.Linear(input_size, hidden_size)])
    #self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_size)])
    self.dropout = nn.ModuleList([nn.Dropout(p=dropout)])
    
    for i in range(num_hidden_layer-1):
      self.lin.append(nn.Linear(hidden_size, hidden_size))
      #self.bn.append(nn.BatchNorm1d(hidden_size))
      self.dropout.append(nn.Dropout(p=dropout))
    
    self.lin.append(nn.Linear(hidden_size, output_size))
    
    if init0:
      for m in self.lin:
        nn.init.constant_(m.weight, 0.)
        nn.init.constant_(m.bias, 0.)


  def forward(self, x):
    input = x
    for i in range(self.num_hidden_layer):
      x = self.lin[i](x)
      #x = self.bn[i](x)
      x = torch.relu(x)
      x = self.dropout[i](x)
    
    x = self.lin[-1](x)

    x = torch.relu(x) + input
    x = torch.sigmoid(x)
    #if self.relu_out:
    #  x = torch.relu(x)
    
    return x


class Sys2LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Sys2LSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    self.lstm = nn.LSTM(input_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
    
    # init biases with 0 except output gate bias negative
    nn.init.constant_(self.lstm.bias_ih_l0, 0.)
    nn.init.constant_(self.lstm.bias_hh_l0, 0.)
    nn.init.constant_(self.lstm.bias_ih_l0[3*hidden_size:], -3.)
    nn.init.constant_(self.lstm.bias_ih_l0[hidden_size:2*hidden_size], 1e+30)
    nn.init.constant_(self.out.bias,  0.)
  
  
  def init_hidden(self, batch_size, device, dtype):
    return torch.zeros(2, 1, batch_size, self.hidden_size).to(device, dtype=dtype)
  
  
  def forward(self, input, state):
    hidden, state = self.lstm(input.view(1, *input.shape), state)
    return self.out(hidden[0]), state


def make_sys2net(input_size, hidden_size, output_size):
  return Sys2LSTM(input_size, hidden_size, output_size)


def make_sys2feat(input_size, hidden_size, output_size, num_hidden_layer, relu_out, init0, dropout):
  return Sys2Net2(input_size, hidden_size, output_size, num_hidden_layer, relu_out, init0, dropout)

