import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True

from unifan.networks import Encoder, Decoder, Set2Gene


class AnnoCluster(nn.Module):

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

    def __init__(self, input_dim: int = 10000, z_dim: int = 32, gene_set_dim: int = 335,
                 tau: float = 1.0, zeta: float = 1.0, n_clusters: int = 16,
                 encoder_dim: int = 128, emission_dim: int = 128, num_layers_encoder: int = 1,
                 num_layers_decoder: int = 1, dropout_rate: float = 0.1, use_t_dist: bool = True,
                 reconstruction_network: str = "gaussian", decoding_network: str = "gaussian",
                 regulating_probability: str = "classifier", centroids: torch.Tensor = None,
                 gene_set_table: torch.Tensor = None, use_cuda: bool = False):

        super().__init__()

        # initialize parameters
        self.z_dim = z_dim
        self.reconstruction_network = reconstruction_network
        self.decoding_network = decoding_network
        self.tau = tau
        self.zeta = zeta
        self.n_clusters = n_clusters
        self.use_t_dist = use_t_dist
        self.regulating_probability = regulating_probability

        if regulating_probability not in ["classifier"]:
            raise NotImplementedError(f"The current implementation only support 'classifier', "
                                      f" for regulating probability.")

        # initialize centroids embeddings
        if centroids is not None:
            self.embeddings = nn.Parameter(centroids, requires_grad=True)
        else:
            self.embeddings = nn.Parameter(torch.randn(self.n_clusters, self.z_dim) * 0.05, requires_grad=True)

        # initialize loss
        self.mse_loss = nn.MSELoss()
        self.nLL_loss = nn.NLLLoss()

        # instantiate encoder for z
        if self.reconstruction_network == "gaussian":
            self.encoder = Encoder(input_dim, z_dim, num_layers=num_layers_encoder, hidden_dim=encoder_dim,
                                   dropout_rate=dropout_rate)
        else:
            raise NotImplementedError(f"The current implementation only support 'gaussian' for encoder.")

        # instantiate decoder for emission
        if self.decoding_network == 'gaussian':
            self.decoder_e = Decoder(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)
            self.decoder_q = Decoder(z_dim, input_dim, num_layers=num_layers_decoder,hidden_dim=emission_dim)
        elif self.decoding_network == 'geneSet':
            self.decoder_e = Set2Gene(gene_set_table)
            self.decoder_q = Set2Gene(gene_set_table)
        else:
            raise NotImplementedError(f"The current implementation only support 'gaussian', "
                                      f"'geneSet' for emission decoder.")

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def forward(self, x):

        # get encoding
        z_e, _ = self.encoder(x)

        # get the index of embedding closed to the encoding
        k, z_dist, dist_prob = self._get_clusters(z_e)

        # get embeddings (discrete representations)
        z_q = self._get_embeddings(k)

        # decode embedding (discrete representation) and encoding
        x_q, _ = self.decoder_q(z_q)
        x_e, _ = self.decoder_e(z_e)

        return x_e, x_q, z_e, z_q, k, z_dist, dist_prob

    def _get_clusters(self, z_e):
        """

        Assign each sample to a cluster based on euclidean distances.

        Parameters
        ----------
        z_e: torch.Tensor
            low-dimensional encodings

        Returns
        -------
        k: torch.Tensor
            cluster assignments
        z_dist: torch.Tensor
            distances between encodings and centroids
        dist_prob: torch.Tensor
            probability of closeness of encodings to centroids transformed by t-distribution

        """

        _z_dist = (z_e.unsqueeze(1) - self.embeddings.unsqueeze(0)) ** 2
        z_dist = torch.sum(_z_dist, dim=-1)

        if self.use_t_dist:
            dist_prob = self._t_dist_sim(z_dist, df=10)
            k = torch.argmax(dist_prob, dim=-1)
        else:
            k = torch.argmin(z_dist, dim=-1)
            dist_prob = None

        return k, z_dist, dist_prob

    def _t_dist_sim(self, z_dist, df=10):
        """
        Transform distances using t-distribution kernel.

        Parameters
        ----------
        z_dist: torch.Tensor
            distances between encodings and centroids

        Returns
        -------
        dist_prob: torch.Tensor
            probability of closeness of encodings to centroids transformed by t-distribution

        """

        _factor = - ((df + 1) / 2)
        dist_prob = torch.pow((1 + z_dist / df), _factor)
        dist_prob = dist_prob / dist_prob.sum(axis=1).unsqueeze(1)

        return dist_prob

    def _get_embeddings(self, k):
        """

        Get the embeddings (discrete representations).

        Parameters
        ----------
        k: torch.Tensor
            cluster assignments

        Returns
        -------
        z_q: torch.Tensor
            low-dimensional embeddings (discrete representations)

        """

        k = k.long()
        _z_q = []
        for i in range(len(k)):
            _z_q.append(self.embeddings[k[i]])

        z_q = torch.stack(_z_q)

        return z_q


    def _loss_reconstruct(self, x, x_e, x_q):
        """
        Calculate reconstruction loss.

        Parameters
        -----------
        x: torch.Tensor
            original observation in full-dimension
        x_e: torch.Tensor
            reconstructed observation encodings
        x_q: torch.Tensor
            reconstructed observation from  embeddings (discrete representations)
        """

        l_e = self.mse_loss(x, x_e)
        l_q = self.mse_loss(x, x_q)
        mse_l = l_e + l_q * self.zeta
        return mse_l

    def _loss_z_prob(self, z_dist, prior_prob=None):
        """
        Calculate annotator loss.

        Parameters
        ----------
        z_dist: torch.Tensor
            distances between encodings and centroids
        prior_prob: torch.Tensor
            probability learned from other source (e.g. prior) about cluster assignment

        """
        if self.regulating_probability == "classifier":

            weighted_z_dist_prob = z_dist * prior_prob
            prob_z_l = torch.mean(weighted_z_dist_prob)

        else:
            raise NotImplementedError(f"The current implementation only support "
                                      f"'classifier' for prob_z_l method.")

        return prob_z_l

    def loss(self, x, x_e, x_q, z_dist, prior_prob=None):

        mse_l = self._loss_reconstruct(x, x_e, x_q)
        prob_z_l = self._loss_z_prob(z_dist, prior_prob)

        l = mse_l + self.tau * prob_z_l

        return l


