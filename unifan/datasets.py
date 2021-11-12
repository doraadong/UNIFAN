import scanpy as sc
import numpy as np
from torch.utils.data import Dataset


class AnnDataset(Dataset):
    def __init__(self, filepath: str, label_name: str = None, second_filepath: str = None,
                 variable_gene_name: str = None):
        """

        Anndata dataset.

        Parameters
        ----------
        label_name: string
            name of the cell type annotation, default 'label'
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """

        super().__init__()

        self.data = sc.read(filepath, dtype='float64', backed="r")

        genes = self.data.var.index.values
        self.genes_upper = [g.upper() for g in genes]
        if label_name is not None:
            self.clusters_true = self.data.obs[label_name].values
        else:
            self.clusters_true = None

        self.N = len(self.data.shape[0])
        self.G = len(self.genes_upper)

        self.secondary_data = None
        if second_filepath is not None:
            self.secondary_data = np.load(second_filepath)
            assert len(self.secondary_data) == self.N, "The other file have same length as the main"

        if variable_gene_name is not None:
            _idx = np.where(self.data.var[variable_gene_name].values)[0]
            self.exp_variable_genes = self.data.X[:, _idx]
            self.variable_genes_names = self.data.var.index.values[_idx]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        main = self.data[idx].X.flatten()

        if self.secondary_data is not None:
            secondary = self.secondary_data[idx].flatten()
            return main, secondary
        else:
            return main


class NumpyDataset(Dataset):
    def __init__(self, filepath: str, second_filepath: str = None):
        """

        Numpy array dataset.

        Parameters
        ----------
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """
        super().__init__()

        self.data = np.load(filepath)
        self.N = self.data.shape[0]
        self.G = self.data.shape[1]

        self.secondary_data = None
        if second_filepath is not None:
            self.secondary_data = np.load(second_filepath)
            assert len(self.secondary_data) == self.N, "The other file have same length as the main"

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        main = self.data[idx].flatten()

        if self.secondary_data is not None:
            secondary = self.secondary_data[idx].flatten()
            return main, secondary
        else:
            return main
