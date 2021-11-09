#!/usr/bin/env python
import os, sys
import argparse
import time
from os.path import exists
import collections
from typing import Iterable
import pickle
from collections import Counter
import gc
import itertools

import torch
import torch.nn as nn
import scanpy as sc
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import umap
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from networkx import *

from context import unifan
from unifan.datasets import AnnDataset, NumpyDataset
from unifan.annocluster import AnnoCluster
from unifan.autoencoder import autoencoder
from unifan.classifier import classifier
from unifan.utils import getGeneSetMatrix
from unifan.trainer import Trainer

prior_name = sys.argv[1]  # pathway.gmt / TF-DNA / no / pathway.gmt+TF-DNA / pathway.gmt+pathway.gmt
z_encoder_layers = int(sys.argv[2])
z_decoder_layers = int(sys.argv[3])
z_encoder_dim = int(sys.argv[4])
z_decoder_dim = int(sys.argv[5])
r_encoder_layers = int(sys.argv[6])
r_decoder_layers = int(sys.argv[7])
r_encoder_dim = int(sys.argv[8])
r_decoder_dim = int(sys.argv[9])
tau = int(sys.argv[12])
tissue = sys.argv[13]
data_name = sys.argv[14]
alpha = float(sys.argv[16])
beta = float(sys.argv[17])
rnetwork = sys.argv[18]
num_epochs_classifier = int(sys.argv[19])
weight_decay = float(sys.argv[21])
features_type = sys.argv[22]
random_seed = int(sys.argv[24])

num_epochs_r = 70
num_epochs_z = 50
r_epoch = num_epochs_r - 1
z_epoch = num_epochs_z - 1

num_epochs_annocluster = 25
z_dim = 32

use_cuda = False
num_workers = 8
batch_size = 128

input_path = "/home/dongshul/projects/lab/anno/input/pbmc/"
output_path = "/home/dongshul/projects/lab/anno/input/init/"
data_filename = f"{tissue}_processed.h5ad"
label_name = "label"
variable_gene_name = "highly_variable"

# ------ training conditions
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    pin_memory = True
    non_blocking = True
else:
    pin_memory = False
    non_blocking = False


def getOutputName(_name):
    if _name[-3:] == 'gmt':
        output_name = _name.split('.')[2]
    else:
        output_name = _name

    return output_name


if '+' in prior_name:
    prior_names_list = prior_name.split('+')
    output_names = []
    for _name in prior_names_list:
        output_names.append(getOutputName(_name))

    output_name = "+".join(output_names)
else:
    output_name = getOutputName(prior_name)

# ------ load data
filepath = os.path.join(input_path, data_filename)
if features_type in ["gene", "gene_gene_sets"]:
    expression_only = AnnDataset(filepath, label_name=label_name, variable_gene_name=variable_gene_name)
    exp_variable_genes = expression_only.exp_variable_genes
    variable_genes_names = expression_only.variable_genes_names
else:
    expression_only = AnnDataset(filepath, label_name=label_name)
    exp_variable_genes = None
    variable_genes_names = None

genes_upper = expression_only.genes_upper
clusters_true = expression_only.clusters_true
N = expression_only.N
G = expression_only.G

# load subset index for calculating ARI
if N > 5e4:
    idx_stratified, _ = train_test_split(range(N), test_size=0.5, stratify=clusters_true)
else:
    idx_stratified = range(N)

# ------ process prior data
# generate gene_set_matrix
if '+' in prior_name:
    _matrix_list = []
    _keys_list = []
    for _name in prior_names_list:
        _matrix, _keys = getGeneSetMatrix(_name, genes_upper)
        _matrix_list.append(_matrix)
        _keys_list.append(_keys)

    gene_set_matrix = np.concatenate(_matrix_list, axis=0)
    keys_all = list(itertools.chain(*_keys_list))

    del _matrix_list
    del _keys_list
    gc.collect()

else:
    gene_set_matrix, keys_all = getGeneSetMatrix(prior_name, genes_upper)

# ------ set-up for the set cover loss
if beta != 0:
    # get the gene set matrix with only genes covered
    genes_covered = np.sum(gene_set_matrix, axis=0)
    gene_covered_matrix = gene_set_matrix[:, genes_covered != 0]
    gene_covered_matrix = torch.from_numpy(gene_covered_matrix).to(device, non_blocking=non_blocking).float()
    beta_list = torch.from_numpy(np.repeat(beta, gene_covered_matrix.shape[1])).to(device,
                                                                           non_blocking=non_blocking).float()

    del genes_covered
    gc.collect()
else:
    gene_covered_matrix = None
    beta_list = None

gene_set_dim = gene_set_matrix.shape[0]
gene_set_matrix = torch.from_numpy(gene_set_matrix).to(device, non_blocking=non_blocking)

# ------ prepare for output
output_parent_path = f"{output_path}{data_name}/{tissue}/"

r_folder = f"{output_parent_path}r"
input_r_ae_path = os.path.join(r_folder, f"r_model_{r_epoch}.pickle")
input_r_path = os.path.join(r_folder, f"r_{r_epoch}.npy")
input_r_names_path = os.path.join(r_folder, f"r_names_{r_epoch}.npy")

pretrain_z_folder = f"{output_parent_path}pretrain_z"
input_z_path = os.path.join(pretrain_z_folder, f"pretrain_z_{z_epoch}.npy")
input_ae_path = os.path.join(pretrain_z_folder, f"pretrain_z_model_{z_epoch}.pickle")
input_cluster_path = os.path.join(pretrain_z_folder, f"cluster_{z_epoch}.npy")

pretrain_annotator_folder = f"{output_parent_path}pretrain_annotator"
annocluster_folder = f"{output_parent_path}annocluster_{features_type}"

# ------ Train gene set activity scores (r) model ------
if features_type == "gene":
    z_gene_set = exp_variable_genes
    set_names = list(variable_genes_names)
else:

    model_gene_set = autoencoder(input_dim=G, z_dim=gene_set_dim, gene_set_dim=gene_set_dim,
                                 encoder_dim=r_encoder_dim, emission_dim=r_decoder_dim,
                                 num_layers_encoder=r_encoder_layers, num_layers_decoder=r_decoder_layers,
                                 reconstruction_network=rnetwork, decoding_network='geneSet',
                                 gene_set_table=gene_set_matrix, use_cuda=use_cuda)

    if os.path.isfile(input_r_path):
        print(f"Inferred r exists. No need to train the gene set activity scores model.")
        z_gene_set = np.load(input_r_path)
    else:
        if os.path.isfile(input_r_ae_path):
            model_gene_set.load_state_dict(torch.load(input_r_ae_path)['state_dict'])

        trainer = Trainer(dataset=expression_only, model=model_gene_set, model_name="r", batch_size=batch_size,
                          num_epochs=num_epochs_r, save_infer=True, output_folder=r_folder, num_workers=num_workers,
                          use_cuda=use_cuda)
        if os.path.isfile(input_r_ae_path):
            print(f"Inferred r model exists but r does not. Need to infer r and no need to train the gene set activity "
                  f"scores model.")
            z_gene_set = trainer.infer_r(alpha=alpha, beta=beta, beta_list=beta_list,
                                         gene_covered_matrix=gene_covered_matrix)
            np.save(input_r_path, z_gene_set)
        else:
            print(f"Start training the gene set activity scores model ... ")
            trainer.train(alpha=alpha, beta=beta, beta_list=beta_list, gene_covered_matrix=gene_covered_matrix)
            z_gene_set = np.load(input_r_path)

    z_gene_set = torch.from_numpy(z_gene_set)

    # filter r to keep only non-zero values
    idx_non_0_gene_sets = np.where(z_gene_set.numpy().sum(axis=0) != 0)[0]

    # get kepted gene set names
    set_names = np.array(keys_all)[idx_non_0_gene_sets]

    z_gene_set = z_gene_set[:, idx_non_0_gene_sets]
    print(f"Aftering filtering, we have {z_gene_set.shape[1]} genesets")

    # add also selected genes if using "gene_gene_sets"
    if features_type == "gene_gene_sets":
        z_gene_set = np.concatenate([z_gene_set, exp_variable_genes], axis=1)
        set_names = list(set_names) + list(variable_genes_names)
    else:
        pass

print(f"z_gene_set: {features_type}: {z_gene_set.shape}")
print(f"z_gene_set: {features_type}: {len(set_names)}")

# save feature names
input_r_names_path = f"{input_r_names_path}_filtered_{features_type}.npy"
np.save(input_r_names_path, set_names)

# save new features
input_r_path = f"{input_r_path}_filtered_{features_type}.npy"
np.save(input_r_path, z_gene_set)
gene_set_dim = z_gene_set.shape[1]

try:
    z_gene_set = z_gene_set.numpy()
except AttributeError:
    pass

# ------ Pretrain annocluster & initialize clustering ------
model_autoencoder = autoencoder(input_dim=G, z_dim=z_dim, gene_set_dim=gene_set_dim,
                                encoder_dim=z_encoder_dim, emission_dim=z_decoder_dim,
                                num_layers_encoder=z_encoder_layers, num_layers_decoder=z_decoder_layers,
                                reconstruction_network='gaussian', decoding_network='gaussian',
                                use_cuda=use_cuda)

if os.path.isfile(input_z_path) and os.path.isfile(input_ae_path):
    print(f"Both pretrained autoencoder and inferred z exist. No need to pretrain the annocluster model.")
    z_init = np.load(input_z_path)
    model_autoencoder.load_state_dict(torch.load(input_ae_path)['state_dict'])
else:
    if os.path.isfile(input_ae_path):
        model_autoencoder.load_state_dict(torch.load(input_ae_path)['state_dict'])

    trainer = Trainer(dataset=expression_only, model=model_autoencoder, model_name="pretrain_z", batch_size=batch_size,
                      num_epochs=num_epochs_z, save_infer=True, output_folder=pretrain_z_folder, num_workers=num_workers,
                      use_cuda=use_cuda)

    if os.path.isfile(input_ae_path):
        print(f"Only pretrained autoencoder exists. Need to infer z and no need to pretrain the annocluster model.")
        z_init = trainer.infer_z()
        np.save(input_z_path, z_init)
    else:
        print(f"Start training pretrain the annocluster model ... ")
        trainer.train()
        z_init = np.load(input_z_path)

z_init = torch.from_numpy(z_init)

try:
    z_init = z_init.numpy()
except AttributeError:
    pass

# initialize using leiden clustering
adata = sc.AnnData(X=z_init)
adata.obsm['X_unifan'] = z_init
sc.pp.neighbors(adata, n_pcs=z_dim,  use_rep='X_unifan', random_state=random_seed)
sc.tl.leiden(adata, resolution=1, random_state=random_seed)
clusters_pre = adata.obs['leiden'].astype('int').values  # original as string

# save for the dataset for classifier training
np.save(input_cluster_path, clusters_pre)

# initialize centroids
try:
    df_cluster = pd.DataFrame(z_init.detach().cpu().numpy())
except AttributeError:
    df_cluster = pd.DataFrame(z_init)

cluster_labels = np.unique(clusters_pre)
M = len(set(cluster_labels))  # set as number of clusters
df_cluster['cluster'] = clusters_pre

# get centroids
centroids = df_cluster.groupby('cluster').mean().values
centroids_torch = torch.from_numpy(centroids)

# ------ pretrain annotator (classification) ------
cls_times = 1  # count how many times of running classification
cls_training_accuracy = 1  # initialize being 1 so that to run at least once
weight_decay_candidates = [50, 20, 10, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1, 5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
idx_starting_weight_decay = weight_decay_candidates.index(weight_decay)

while cls_training_accuracy >= 0.99:

    # assign new weight decay (first time running kepted the same)
    weight_decay = weight_decay_candidates[idx_starting_weight_decay - cls_times + 1]

    print(f"Run classifier the {cls_times}th time with {weight_decay}")

    prior_cluster = NumpyDataset(input_r_path, input_cluster_path)

    model_classifier = classifier(output_dim=M, z_dim=gene_set_dim, emission_dim=128, use_cuda=use_cuda)

    trainer = Trainer(dataset=prior_cluster, model=model_classifier, model_name="pretrain_annotator", batch_size=32,
                      num_epochs=num_epochs_classifier, save_infer=False, output_folder=pretrain_annotator_folder,
                      num_workers=num_workers, use_cuda=use_cuda)

    trainer.train(weight_decay=weight_decay)
    clusters_classifier = trainer.infer_z()

    cls_training_accuracy = (clusters_classifier.numpy() == clusters_pre).sum() / N
    print(f"Cluster accuracy on training: \n {cls_training_accuracy}")

    cls_times += 1

# ------ clustering ------
num_epochs = num_epochs_annocluster 
use_pretrain = True

model_annocluster = AnnoCluster(input_dim=G, z_dim=z_dim, gene_set_dim=gene_set_dim, tau=tau, n_clusters=M,
                                encoder_dim=z_encoder_dim, emission_dim=z_decoder_dim, 
                                num_layers_encoder=z_encoder_layers, num_layers_decoder=z_decoder_layers, 
                                use_t_dist=True, reconstruction_network='gaussian', decoding_network='gaussian', 
                                centroids=centroids_torch, gene_set_table=gene_set_matrix, use_cuda=use_cuda)

if use_pretrain:
    pretrained_state_dict = model_autoencoder.state_dict()

    # re-initialize parameters related to clusters
    state_dict = model_annocluster.state_dict()
    for k, v in state_dict.items():
        if k in pretrained_state_dict.keys():
            state_dict[k] = pretrained_state_dict[k]

    model_annocluster.load_state_dict(state_dict)

# reload dataset, loading gene set activity scores together
filepath = os.path.join(input_path, data_filename)
expression_prior = AnnDataset(filepath, second_filepath=input_r_path, label_name=label_name)

trainer = Trainer(dataset=expression_prior, model=model_annocluster, model_2nd=model_classifier,
                  model_name="autocluster", batch_size=batch_size, num_epochs=num_epochs_annocluster,
                  save_infer=True, output_folder=annocluster_folder, num_workers=num_workers, use_cuda=use_cuda)
trainer.train(weight_decay=weight_decay)




