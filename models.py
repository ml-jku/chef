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



