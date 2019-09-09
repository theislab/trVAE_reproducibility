import anndata

from reptrvae.loader import Loader
from reptrvae.models._network import Network
from reptrvae.utils import remove_sparsity
from ._saucie_backend import SAUCIE_BACKEND


class SAUCIE(Network):
    def __init__(self, x_dimension, **kwargs):
        super().__init__()
        self.x_dimension = x_dimension

        self.model_backend = SAUCIE_BACKEND(x_dimension, **kwargs)

    def __create_network(self):
        pass

    def __compile_network(self):
        pass

    def to_latent(self, adata, labels):
        adata = remove_sparsity(adata)
        data_loader = Loader(data=adata.X, labels=labels, shuffle=False)

        latent = self.model_backend.get_embedding(data_loader)
        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)
        return latent_adata

    def predict(self):
        pass

    def train(self):
        pass
