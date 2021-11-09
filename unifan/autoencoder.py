import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True

from unifan.networks import Encoder, Decoder, Set2Gene, LinearCoder, NonNegativeCoder, SigmoidCoder


class autoencoder(nn.Module):
    """

    Autoencoder used for pre-training.

    Parameters
    ----------
    input_dim: integer
        number of input features
    z_dim: integer
        number of low-dimensional features
    gene_set_dim: integer
        number of gene sets
    encoder_dim: integer
        dimension of hidden layer for encoders
    emission_dim: integer
        dimension of hidden layer for decoders
    num_layers_encoder: integer
        number of hidden layers  for encoder
    num_layers_decoder: integer
        number of hidden layers  for decoder
    dropout_rate: float
    gene_set_table: torch.Tensor
        gene set relationship table

    """

    def __init__(self, input_dim: int = 10000, z_dim: int = 32, gene_set_dim: int = 335, encoder_dim: int = 128,
                 emission_dim: int = 128, num_layers_encoder: int = 1, num_layers_decoder: int = 1,
                 dropout_rate: float = 0.1, reconstruction_network: str = "non-negative",
                 decoding_network: str = "geneSet", gene_set_table: torch.Tensor = None, use_cuda: bool = False):

        super().__init__()

        # initialize parameters
        self.z_dim = z_dim
        self.reconstruction_network = reconstruction_network
        self.decoding_network = decoding_network

        # initialize loss
        self.mse_loss = nn.MSELoss()

        # initialize encoder and decoder
        if self.reconstruction_network == 'linear' and self.decoding_network == 'linear':
            self.encoder = LinearCoder(input_dim, z_dim)
            self.decoder_e = LinearCoder(z_dim, input_dim)
        else:

            if self.reconstruction_network == 'non-negative':
                # instantiate encoder for z
                self.encoder = NonNegativeCoder(input_dim, z_dim, num_layers=num_layers_encoder, hidden_dim=encoder_dim,
                                                dropout_rate=dropout_rate)
            elif self.reconstruction_network == 'sigmoid':
                # instantiate encoder for z
                self.encoder = SigmoidCoder(input_dim, z_dim, num_layers=num_layers_encoder, hidden_dim=encoder_dim,
                                            dropout_rate=dropout_rate)
            elif self.reconstruction_network == "gaussian":
                # instantiate encoder for z, using standard encoder
                self.encoder = Encoder(input_dim, z_dim, num_layers=num_layers_encoder, hidden_dim=encoder_dim,
                                       dropout_rate=dropout_rate)

            else:
                raise NotImplementedError(f"The current implementation only support 'gaussian', "
                                          f"'non-negative' or 'sigmoid' for encoder.")

            # instantiate decoder for emission
            if self.decoding_network == 'gaussian':
                self.decoder_e = Decoder(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)
            elif self.decoding_network == 'geneSet':
                self.decoder_e = Set2Gene(gene_set_table)

            else:
                raise NotImplementedError(f"The current implementation only support 'gaussian', "
                                          f"'geneSet' for emission decoder.")

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def forward(self, data):

        x = data

        # get encoding
        z_e, _ = self.encoder(x)

        # decode encoding
        x_e, _ = self.decoder_e(z_e)

        return x_e, z_e

    def _loss_reconstruct(self, x, x_e):
        """
        Calculate reconstruction loss.

        Parameters
        ----------
        x: torch.Tensor
            original data
        x_e: torch.Tensor
            reconstructed data

        Returns
        -------
        mse_l: torch.Tensor
            reconstruction loss

        """
        l_e = self.mse_loss(x, x_e)
        mse_l = l_e

        return mse_l

    def loss(self, x, x_e):
        l = self._loss_reconstruct(x, x_e)
        return l
