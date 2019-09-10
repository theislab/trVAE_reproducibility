import anndata
import numpy as np
from sklearn.preprocessing import LabelEncoder

from reptrvae.loader import Loader
from reptrvae.models._network import Network
from reptrvae.utils import remove_sparsity, label_encoder
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
        latent = latent[0]
        latent = np.nan_to_num(latent)
        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)
        return latent_adata

    def to_mmd_layer(self, adata, labels):
        adata = remove_sparsity(adata)

        data_loader = Loader(data=adata.X, labels=labels, shuffle=False)

        latent = self.model_backend.get_layer(data_loader, 'mmd')
        latent = latent[0]
        latent = np.nan_to_num(latent)
        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata

    def predict(self, adata, target_label, condition_key, cell_type_key, cell_type_to_predict, source_condition,
                target_condition):
        adata = remove_sparsity(adata)

        cell_type_adata = adata[adata.obs[cell_type_key] == cell_type_to_predict]
        source_adata = cell_type_adata[cell_type_adata.obs[condition_key] == source_condition]

        y_test = np.zeros(source_adata.shape[0]) + target_label
        real_loader = Loader(source_adata.X, labels=y_test, shuffle=False)

        pred = self.model_backend.get_reconstruction(real_loader)
        pred = np.nan_to_num(pred[0])

        pred_adata = anndata.AnnData(X=pred)
        pred_adata.obs[condition_key] = f"{cell_type_to_predict}_pred_{target_condition}"
        pred_adata.var_names = adata.var_names

        return pred_adata

    def train(self, train_adata, condition_key, le=None, n_epochs=1000, batch_size=256):
        train_adata = remove_sparsity(train_adata)

        x_train = train_adata.X
        y_train, _ = label_encoder(train_adata, le, condition_key)
        y_train = np.reshape(y_train, (-1,))

        train_loader = Loader(x_train, labels=y_train, shuffle=True)

        self.model_backend.train(train_loader, n_epochs, batch_size)
