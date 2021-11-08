
import os
import argparse
import time
from os.path import exists
import collections
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.autograd import Function

class FullyConnectedLayers(nn.Module):
    """
    Parameters
    ----------
    input_dim: integer
        number of input features
    output_dim: integer
        number of output features
    num_layers: integer
        number of hidden layers
    hidden_dim: integer
        dimension of hidden layer
    dropout_rate: float
    bias: boolean
        if apply bias to the linear layers
    batch_norm: boolean
        if apply batch normalization

    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, hidden_dim: int = 128,
                 dropout_rate: float = 0.1, bias: bool = True, batch_norm: bool = False):
        super().__init__()
        layers_dim = [input_dim] + [hidden_dim for i in range(num_layers - 1)] + [output_dim]

        self.all_layers = nn.Sequential(collections.OrderedDict(
            [(f"Layer {i}", nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=bias),
                nn.BatchNorm1d(output_dim) if batch_norm else None,
                nn.ReLU(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None))
             for i, (input_dim, output_dim) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))]))

    def forward(self, x: torch.Tensor):

        for layers in self.all_layers:
            for layer in layers:
                if layer is not None:
                    x = layer(x)

        return x


class Encoder(nn.Module):
    """

    A standard encoder.

    Parameters
    ----------
    input_dim: integer
        number of input features
    output_dim: integer
        number of output features
    num_layers: integer
        number of hidden layers
    hidden_dim: integer
        dimension of hidden layer
    dropout_rate: float

    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, hidden_dim: int = 128, 
                 dropout_rate: float = 0.1):
        super().__init__()

        self.encoder = FullyConnectedLayers(input_dim=input_dim, output_dim=hidden_dim, num_layers=num_layers,
                                hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.var_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        q_m: torch.Tensor
            estimated mean
        q_v: torch.Tensor
            estimated variance

        """

        q = self.encoder(x)
        q_m = self.mean_layer(q)
        q_v = torch.exp(self.var_layer(q))
        return q_m, q_v


class LinearCoder(nn.Module):

    """

    A single-layer linear encoder.

    Parameters
    ----------
    input_dim: integer
        number of input features
    output_dim: integer
        number of output features
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.encoder = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        q = self.encoder(x)
        return q, None


class NonNegativeCoder(nn.Module):
    
    """

    A encoder outputting non-negative values (using ReLU for the output layer). 

    Parameters
    ----------
    input_dim: integer
        number of input features
    output_dim: integer
        number of output features
    num_layers: integer
        number of hidden layers
    hidden_dim: integer
        dimension of hidden layer
    dropout_rate: float

    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, hidden_dim: int = 128, 
                 dropout_rate: float = 0.1):
        super().__init__()

        self.encoder = FullyConnectedLayers(input_dim=input_dim, output_dim=hidden_dim, num_layers=num_layers,
                                hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.mean_layer = FullyConnectedLayers(input_dim=hidden_dim, output_dim=output_dim, num_layers=1,
                                hidden_dim=hidden_dim, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor):
        q = self.encoder(x)
        q = self.mean_layer(q)
        return q, None


class SigmoidCoder(nn.Module):
    """

    A encoder using sigmoid for the output layer. 

    Parameters
    ----------
    input_dim: integer
        number of input features
    output_dim: integer
        number of output features
    num_layers: integer
        number of hidden layers
    hidden_dim: integer
        dimension of hidden layer
    dropout_rate: float

    """
    
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, hidden_dim: int = 128, 
                 dropout_rate: float = 0.1):
        super().__init__()

        self.encoder = FullyConnectedLayers(input_dim=input_dim, output_dim=hidden_dim, num_layers=num_layers,
                                hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.mean_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        q = self.encoder(x)
        q = self.mean_layer(q)
        return q, None


class Decoder(nn.Module):
    """

    A standard decoder.

    Parameters
    ----------
    input_dim: integer
        number of input features
    output_dim: integer
        number of output features
    num_layers: integer
        number of hidden layers
    hidden_dim: integer
        dimension of hidden layer
    dropout_rate: float

    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, hidden_dim: int = 128, 
                 dropout_rate: float = 0.1):
        super().__init__()
        self.decoder = FullyConnectedLayers(input_dim=input_dim, output_dim=hidden_dim, num_layers=num_layers,
                                hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.var_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        p_m: torch.Tensor
            estimated mean
        p_v: torch.Tensor
            estimated variance
        """

        p = self.decoder(x)
        p_m = self.mean_layer(p)
        p_v = torch.exp(self.var_layer(p))
        return p_m, p_v



class Set2Gene(nn.Module):

    """
    Decode by linear combination of known gene set relationship between gene set (input) and genes (output).

    Parameters
    ----------
    tf_gene_table: torch.Tensor
        number of genes x number gene sets (equal to the dimension of input)

    """

    def __init__(self,  tf_gene_table: torch.Tensor):
        super().__init__()
        self.tf_gene_table = tf_gene_table

    def forward(self, x: torch.Tensor):
        p_m = torch.mm(x.double(), self.tf_gene_table)
        return p_m, None


class Decode2Labels(nn.Module):
    """

    A linear classifier (logistic classifier).

    Parameters
    ----------
    input_dim: integer
        number of input features
    output_dim: integer
        number of output features
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.predictor = nn.Sequential(nn.Linear(input_dim, output_dim, bias=bias), nn.LogSoftmax(dim=-1))

    def forward(self, x: torch.Tensor):
        labels = self.predictor(x)
        return labels

