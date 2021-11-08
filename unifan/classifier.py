import os
import argparse
import time
from os.path import exists
import collections
from typing import Iterable
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vati.networks import Decode2Labels

torch.backends.cudnn.benchmark = True


class classifier(nn.Module):
    """

    A classifier.

    Parameters
    ----------
    z_dim: integer
        number of input features
    output_dim: integer
        number of output (number of types of labels)
    emission_dim: integer
        dimension of hidden layer
    num_layers: integer
        number of hidden layers

    """

    def __init__(self, z_dim: int = 335, output_dim: int = 10, emission_dim: int = 128, num_layers: int = 1,
                 use_cuda=False):

        super().__init__()

        # initialize loss
        self.loss_function = nn.NLLLoss()

        # instantiate decoder for emission
        self.decoder = Decode2Labels(z_dim, output_dim)

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()


    def forward(self, x):
        y_pre = self.decoder(x)
        return y_pre

    def loss(self, y_pre, y_true):
        l = self.loss_function(y_pre, y_true)
        return l
