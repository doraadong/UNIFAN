import os
import argparse
import time
from os.path import exists
import collections
from typing import Iterable
import math

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

class Trainer(nn.Module):

    """
    Clustering with annotator.

    Parameters
    ----------
    input_dim: integer
        number of input features
    z_dim: integer
        number of low-dimensional features
    gene_set_dim: integer
        number of gene sets
    tau: float
        hyperparameter to weight the annotator loss
    zeta: float
        hyperparameter to weight the reconstruction loss from embeddings (discrete representations)
    encoder_dim: integer
        dimension of hidden layer for encoders
    emission_dim: integer
        dimension of hidden layer for decoders
    num_layers_encoder: integer
        number of hidden layers  for encoder
    num_layers_decoder: integer
        number of hidden layers  for decoder
    dropout_rate: float
    use_t_dist: boolean
        if using t distribution kernel to transform the euclidean distances between encodings and centroids
    regulating_probability: string
        the type of probability to regulating the clustering (by distance) results
    centroids: torch.Tensor
        embeddings in the low-dimensional space for the cluster centroids
    gene_set_table: torch.Tensor
        gene set relationship table

    """

    def __init__(self, dataset, model, model_2nd=None, model_name: str = None, batch_size: int = 128,
                 num_epochs: int = 50, percent_training: float = 1.0, learning_rate: float = 0.0005,
                 decay_factor: float = 0.9, num_workers: int = 8, use_cuda: bool = False, checkpoint_freq: int = 20,
                 val_freq: int = 10, visualize_freq: int = 10, save_visual: bool = False, save_checkpoint: bool = False,
                 save_infer: bool = False, output_folder: str = None):

        super().__init__()

        # device
        self.num_workers = num_workers
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.pin_memory = True if use_cuda else False
        self.non_blocking = True if use_cuda else False

        # model
        _support_models = ["r", "pretrian_z", "annocluster", "pretrain_annotator"]
        if model_name not in _support_models:
            raise NotImplementedError(f"The current implementation only support training "
                                      f"for {','.join(_support_models)}.")
        self.model_name = model_name
        self.model = model.to(self.device)
        self.model_2nd = model_2nd.to(self.device) if model_2nd is not None else None

        # optimization
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, self.decay_factor)
        if model_2nd is not None:
            self.optimizer_2nd = torch.optim.Adam(self.model_2nd.parameters(), lr=self.learning_rate)
            self.scheduler_2nd = torch.optim.lr_scheduler.StepLR(self.optimizer_2nd, 1000, self.decay_factor)
        else:
            self.optimizer_2nd, self.scheduler_2nd = None, None

        # evaluation
        self.checkpoint_freq = checkpoint_freq
        self.val_frequency = val_freq
        self.visual_frequency = visualize_freq
        self.output_folder = output_folder
        self.percent_training = percent_training
        self.save_checkpoint = save_checkpoint
        self.save_visual = save_visual
        self.save_infer = save_infer

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # data
        self.dataset = dataset
        # prepare for data loader
        train_length = int(self.dataset.N * self.percent_training)
        val_length = self.dataset.N - train_length

        train_data, val_data = torch.utils.data.random_split(self.dataset, (train_length, val_length))
        self.dataloader_all = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.percent_training != 1:
            self.dataloader_val = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True,
                                                         num_workers=self.num_workers, pin_memory=self.pin_memory)
        else:
            self.dataloader_val = None

        # initialize functions for each model
        self.train_functions = {"r": self.train_epoch_r, "pretrain_z": self.train_epoch_z,
                                       "pretrain_annotator": self.train_epoch_annotator,
                                       "annocluster": self.train_epoch_cluster}
        self.train_stats_dicts = {"r": {k: [] for k in ['train_loss', 'val_loss', 'train_sparsity', 'train_set_cover']},
                                  "pretrain_z": {k: [] for k in ['train_loss', 'val_loss']},
                                  "pretrain_annotator": {k: [] for k in ['train_loss', 'val_loss']},
                                  "annocluster": {k: [] for k in ['train_loss', 'val_loss', 'train_mse', 'train_mse_e',
                                                                  'train_mse_q', 'train_prob_z_l', 'ARI', 'NMI']}}

        self.evaluate_functions = {"r": self.evaluate_r, "pretrain_z": self.evaluate_z,
                                "pretrain_annotator": process_minibatch_annotator,
                                "annocluster": process_minibatch_cluster}


        if self.model_name in ["pretrain_annotator", "annocluster"]:
            self.annotator_param_list = nn.ParameterList()
            if self.model_name == "pretrain_annotator":
                for p in self.model.named_parameters():
                    self.annotator_param_list.append(p[1])
            else:
                assert self.model_2nd is not None
                for p in self.model_2nd.named_parameters():
                    self.annotator_param_list.append(p[1])

        if self.model_name == "pretrain_annotator" and self.save_visual:
            raise NotImplementedError(f"The current implementation only support visualizing "
                                      f"for {','.join(_support_models)[:-1]}.")

        if self.model_name != "annocluster" and self.model_2nd is not None:
            raise NotImplementedError(f"The current implementation only support two models training for annocluster")

    def train(self, **kwargs):

        for epoch in range(self.num_epochs):
            print(f"Current epoch: {epoch}")
            self.model.train()
            if self.model_2nd is not None:
                self.model_2nd.train()

            print(f"Start training...")
            self.train_functions[self.model_name](**kwargs)

            if epoch % self.val_freq == 0:
                with torch.no_grad():
                    self.model.eval()
                    if self.model_2nd is not None:
                        self.model_2nd.eval()

                    self.evaluate_functions[self.model_name](**kwargs)

            if self.save_visual and epoch % self.visual_frequency == 0:
                with torch.no_grad():
                    self.model.eval()
                    if self.model_2nd is not None:
                        self.model_2nd.eval()
                    if self.model_name == "annocluster":
                        _X, _clusters = self.infer_functions[self.model_name](**kwargs)
                    else:
                        _X = self.infer_functions[self.model_name](**kwargs)
                        _clusters = None
                    self.visualize_UMAP(_X, epoch, self.dataset.clusters_true, self.output_folder,
                                        clusters_pre=_clusters)

            # save model & inference
            if self.save_checkpoint and epoch % self.checkpoint_freq == 0:
                # save model
                _state = {'epoch': epoch,
                          'state_dict': self.model.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
                if self.model_2nd is not None:
                    _state = {'epoch': epoch,
                              'state_dict': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict(),
                              'state_dict_2': self.model_2nd.state_dict(),
                              'optimizer_2': self.optimizer_2nd.state_dict()}

                # default (no epoch identifier) model are trained with 20 epochs
                torch.save(_state, os.path.join(self.output_folder, f"{self.model_name}_model_{epoch}.pickle"))

                del _state
                gc.collect()

        # save model
        _state = {'epoch': epoch,
                  'state_dict': self.model.state_dict(),
                  'optimizer': self.optimizer.state_dict()}
        if self.model_2nd is not None:
            _state = {'epoch': epoch,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'state_dict_2': self.model_2nd.state_dict(),
                      'optimizer_2': self.optimizer_2nd.state_dict()}

        # default (no epoch identifier) model are trained with 20 epochs
        torch.save(_state, os.path.join(self.output_folder, f"{self.model_name}_model_{epoch}.pickle"))

        del _state
        gc.collect()

        # inference and save
        if self.save_infer:
            if self.model_name == "annocluster":
                _X, _clusters = self.infer_functions[self.model_name](**kwargs)
                np.save(os.path.join(self.output_folder, f"{self.model_name}_clusters_pre_{epoch}.npy"), _clusters)
            else:
                _X = self.infer_functions[self.model_name](**kwargs)

            np.save(os.path.join(self.output_folder, f"{self.model_name}_{epoch}.npy"), _X)

        # save training stats
        with open(os.path.join(self.output_folder, f"stats_{epoch}.pickle"), 'wb') as handle:
            pickle.dump(train_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train_epoch_r(self, **kwargs):
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            self.process_minibatch_r(X_batch=X_batch, **kwargs)

    def train_epoch_z(self,  **kwargs):
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            self.process_minibatch_z(X_batch=X_batch, **kwargs)

    def train_epoch_annotator(self,  **kwargs):
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(self.dataloader_train)):
            y_batch = torch.flatten(y_batch)
            self.process_minibatch_annotator(X_batch, y_batch, **kwargs)

    def train_epoch_cluster(self,  **kwargs):
        for batch_idx, (X_batch, gene_set_batch) in enumerate(tqdm(self.dataloader_train)):
            self.process_minibatch_cluster(X_batch, gene_set_batch, **kwargs)

    def process_minibatch_cluster(self, X_batch, gene_set_batch, weight_decay: float = 0):
        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
        gene_set_batch = gene_set_batch.to(self.device, non_blocking=self.non_blocking).float()

        if self.model.training and self.model_2nd.training:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_2nd.zero_grad(set_to_none=True)

        x_e, x_q, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch)
        y_pre = self.model_2nd(gene_set_batch)
        pre_prob = torch.exp(y_pre)

        l = self.model.loss(X_batch, x_e, x_q, z_dist, prior_prob=pre_prob)
        l_prob = self.model_2nd.loss(y_pre, k)

        l_prob += weight_decay * torch.sum(torch.square(torch.sum(torch.abs(self.annotator_param_list[0]), dim=0)))

        if self.model.training and self.model_2nd.training:
            l.backward(retain_graph=True)
            self.optimizer.step()

            l_prob.backward(retain_graph=True)
            self.optimizer_2nd.step()

            self.scheduler.step()
            self.scheduler_2nd.step()

        return l.detach().item(), l_prob.detach().numpy(), k.detach().numpy(), z_e.detach().numpy()

    def evaluate_cluster(self, **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]

        _loss = []
        _l_e = []
        _l_q = []
        _mse_l = []
        _prob_z_l = []

        for batch_idx, (X_batch, gene_set_batch) in enumerate(tqdm(self.dataloader_train)):
            X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
            gene_set_batch = gene_set_batch.to(self.device, non_blocking=self.non_blocking).float()

            x_e, x_q, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch)
            y_pre = self.model_2nd(gene_set_batch)
            pre_prob = torch.exp(y_pre)

            l = self.model.loss(X_batch, x_e, x_q, z_dist, prior_prob=pre_prob).detach().item()
            l_prob = self.model_2nd.loss(y_pre, k).detach().item()

            # calculate each loss
            l_e = self.model.mse_loss(X_batch, x_e).detach().item()
            l_q = self.model.mse_loss(X_batch, x_q).detach().item()
            mse_l = self.model._loss_reconstruct(X_batch, x_e, x_q).detach().item()
            prob_z_l = self.model._loss_z_prob(z_dist, prior_prob=pre_prob).detach().item()

            _loss.append(l)
            _l_e.append(l_e)
            _l_q.append(l_q)
            _mse_l.append(mse_l)
            _prob_z_l.append(prob_z_l)

        train_stats['train_loss'].append(np.mean(_loss))
        train_stats['train_mse'].append(np.mean(_mse_l))
        train_stats['train_mse_e'].append(np.mean(_l_e))
        train_stats['train_mse_q'].append(np.mean(_l_q))
        train_stats['train_prob_z_l'].append(np.mean(_prob_z_l))

        print(f"Start evaluating val...")
        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch, gene_set_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _, _, _ = self.process_minibatch_cluster(X_batch, gene_set_batch, **kwargs)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(0)

    def infer_cluster(self, **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]

        with torch.no_grad():
            self.model.eval()
            if self.model_2nd is not None:
                self.model_2nd.eval()

            # inference
            k_list = []
            z_e_list = []

            for batch_idx, (X_batch, gene_set_batch) in enumerate(tqdm(self.dataloader_all)):
                _, _, k, z_e = self.process_minibatch_cluster(X_batch, gene_set_batch, **kwargs)
                k_list.append(k)
                z_e_list.append(z_e)

            clusters_pre = np.concatenate(k_list)
            z_annocluster = np.concatenate(z_e_list)

            if self.dataset.N > 5e4:
                idx_stratified, _ = train_test_split(range(self.dataset.N), test_size=0.5,
                                                     stratify=self.dataset.clusters_true)

            # metrics
            ari_smaller = adjusted_rand_score(clusters_pre[idx_stratified],
                                              self.dataset.clusters_true[idx_stratified])
            nmi_smaller = adjusted_mutual_info_score(clusters_pre, self.dataset.clusters_true)
            print(f"annocluster: ARI for smaller cluster: {ari_smaller}")
            print(f"annocluster: NMI for smaller cluster: {nmi_smaller}")

            train_stats["ARI"].append(ari_smaller)
            train_stats["NMI"].append(nmi_smaller)

        return z_annocluster, clusters_pre


    def process_minibatch_annotator(self, X_batch, y_batch, weight_decay: float = 0):

        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
        y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)

        y_pre = self.model(X_batch)
        l = self.model.loss(y_pre, y_batch)

        l += weight_decay * torch.sum(torch.square(torch.sum(torch.abs(self.annotator_param_list[0]), dim=0)))

        if self.model.training:
            l.backward()
            self.optimizer.step()
            self.scheduler.step()

        return l.detach().item(), y_pre.detach().numpy()

    def evaluate_annotator(self,  **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]
        _loss = []
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(self.dataloader_train)):
            y_batch = torch.flatten(y_batch)
            l, _ = self.process_minibatch_annotator(X_batch, y_batch, self.annotator_param_list, **kwargs)
            _loss.append(l)

        train_stats['train_loss'].append(np.mean(_loss))

        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch, y_batch) in enumerate(tqdm(self.dataloader_val)):
                y_batch = torch.flatten(y_batch)
                l, _ = self.process_minibatch_annotator(X_batch, y_batch, self.annotator_param_list, **kwargs)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(0)

    def infer_annotator(self):
        with torch.no_grad():
            self.model.eval()
            tf_prob = self.model(torch.from_numpy(self.dataset.data).float())

        clusters_prob_pre = torch.exp(tf_prob)
        clusters_classifier = np.argmax(clusters_prob_pre, axis=1)

        return clusters_classifier

    def process_minibatch_r(self, X_batch, alpha: float = 0, beta: float = 0, beta_list: torch.Tensor = None,
                            gene_covered_matrix: torch.Tensor = None):
        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)

        x_e, z_e = self.model(X_batch)
        l = self.model.loss(X_batch.float(), x_e.float())

        # add sparsity regularization on the output
        if alpha != 0:
            sparsity_penalty = alpha * torch.mean(torch.abs(z_e))
            l += sparsity_penalty
        else:
            sparsity_penalty = torch.zeros(1)

        if beta != 0:
            cover_penality = - torch.mean(torch.matmul(torch.mm(z_e, gene_covered_matrix),
                                                       beta_list))
            l += cover_penality
        else:
            cover_penality = torch.zeros(1)

        if self.model.training:
            l.backward()
            self.optimizer.step()
            self.scheduler.step()

        return l.detach().cpu().item(), z_e.detach().cpu().numpy(), \
               sparsity_penalty.detach().cpu().numpy(), cover_penality.detach().cpu().numpy()

    def evaluate_r(self, **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]
        _loss = []
        _sp = []
        _cp = []

        print(f"Start evaluating train ...")
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            l, _, sp, cp = self.process_minibatch_r(X_batch, **kwargs)
            _loss.append(l)
            _sp.append(sp)
            _cp.append(cp)

        print(f"Finish evaluating train...")

        train_stats['train_loss'].append(np.mean(_loss))
        train_stats['train_sparsity'].append(np.mean(_sp))
        train_stats['train_set_cover'].append(np.mean(_cp))

        print(f"Start evaluating val...")
        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _, _, _ = self.process_minibatch_r(X_batch, **kwargs)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(0)

        print(f"Finish evaluating val...")

    def infer_r(self, **kwargs):

        """

        Parameters
        ----------
        dataloader_all
        clusters_true

        Returns
        -------

        """
        with torch.no_grad():
            self.model.eval()

            z_e_list = []
            _loss = []
            print(f"Start inferring ...")
            for batch_idx, (X_batch) in enumerate(self.dataloader_all):
                l, z_e, _, _ = self.process_minibatch_r(X_batch, **kwargs)

                z_e_list.append(z_e)
                _loss.append(l)

            print(f"Finish inferring ...")
            z_gene_set = np.concatenate(z_e_list)

            del z_e_list
            del _loss
            del z_e

            gc.collect()

        return z_gene_set

    def process_minibatch_z(self, X_batch):

        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)

        x_e, z_e = self.model(X_batch)

        l = self.model.loss(X_batch.float(), x_e.float())

        if self.model.training:
            l.backward()
            self.optimizer.step()
            self.scheduler.step()
        return l.detach().cpu().item(), z_e.detach().cpu().numpy()

    def evaluate_z(self, **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]

        _loss = []
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            l, _ = self.process_minibatch_z(X_batch)
            _loss.append(l)

        train_stats['train_loss'].append(np.mean(_loss))

        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _ = self.process_minibatch_z(X_batch)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(0)

    def infer_z(self):

        with torch.no_grad():
            self.model.eval()

            z_e_list = []

            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_all)):
                _, z_e = self.process_minibatch_z(X_batch)
                z_e_list.append(z_e)

            z_init = np.concatenate(z_e_list)

            del z_e_list
            gc.collect()

        return z_init

    @staticmethod
    def visualize_UMAP(X, epoch:int, clusters_true, output_folder: str, clusters_pre=None, color_palette:str = "tab20"):
        """
        Parameters
        ----------
        dataloader_all
        clusters_true

        Returns
        -------
        :param output_folder:

        """

        print(f"Start visualizing using UMAP...")
        umap_original = umap.UMAP().fit_transform(X)

        # color by cluster
        if clusters_pre is not None:
            hues = {'label': clusters_true, 'cluster': clusters_pre}
        else:
            hues = {'label': clusters_true}
        for k, v in hues.items():
            df_plot = pd.DataFrame(umap_original)
            df_plot['label'] = v
            df_plot['label'].astype('str')
            df_plot.columns = ['dim_1', 'dim_2', 'label']

            plt.figure(figsize=(10, 10))
            sns.scatterplot(x='dim_1', y='dim_2', hue='label', data=df_plot, palette=color_palette,
                            legend=False)
            plt.title( f"Encoding (r) colored by {k}")
            plt.savefig(os.path.join(output_folder, f"r_{epoch}_{k}.png"), bbox_inches="tight", format="png")
            plt.close()

        del X
        del umap_original
        del df_plot

        gc.collect()
