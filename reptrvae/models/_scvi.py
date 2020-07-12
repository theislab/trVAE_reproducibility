import os

import numpy as np
import scanpy as sc
import torch
from scvi.dataset import AnnDatasetFromAnnData
from scvi.inference import UnsupervisedTrainer
from scvi.models import *
from sklearn.preprocessing import LabelEncoder

from reptrvae.models._network import Network


class scVI(Network):
    def __init__(self, x_dimension, n_batches, **kwargs):
        super().__init__()
        self.x_dimension = x_dimension
        self.n_batches = n_batches
        self.use_batches = True

        self.model_path = kwargs.get("model_path", "./")

        self.__create_network()

    def __create_network(self):
        self.model = VAE(self.x_dimension,
                         reconstruction_loss='nb',
                         n_batch=self.n_batches * self.use_batches,
                         dispersion="gene")

    def __compile_network(self):
        pass

    def to_latent(self, adata, condition_key, cell_type_key):
        le = LabelEncoder()
        adata.obs['labels'] = le.fit_transform(adata.obs[cell_type_key].values)
        adata.obs['batch_indices'] = le.fit_transform(adata.obs[condition_key].values)

        net_adata = AnnDatasetFromAnnData(adata)

        posterior = self.trainer.create_posterior(self.trainer.model, net_adata, indices=np.arange(len(net_adata)))

        latent, _, __ = posterior.sequential().get_latent()

        latent_adata = sc.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)
        return latent_adata

    def to_mmd_layer(self, adata, condition_key, cell_type_key):
        le = LabelEncoder()
        adata.obs['labels'] = le.fit_transform(adata.obs[cell_type_key].values)
        adata.obs['batch_indices'] = le.fit_transform(adata.obs[condition_key].values)

        net_adata = AnnDatasetFromAnnData(adata)

        posterior = self.trainer.create_posterior(self.trainer.model, net_adata, indices=np.arange(len(net_adata)))

        latent, _, __ = posterior.sequential().get_latent()

        latent_adata = sc.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)
        return latent_adata

    def predict(self, adata, cell_type_to_predict, condition_key, cell_type_key, target_condition, source_condition,
                n_generated_samples=50):
        cell_type_adata = adata.copy()[adata.obs[cell_type_key] == cell_type_to_predict]

        real_adata = cell_type_adata[cell_type_adata.obs[condition_key] == target_condition]
        ctrl_adata = cell_type_adata[cell_type_adata.obs[condition_key] == source_condition]

        le = LabelEncoder()
        le.fit([source_condition, target_condition])
        real_adata.obs['batch_indices'] = le.transform(real_adata.obs[condition_key].values)
        ctrl_adata.obs['batch_indices'] = le.transform([target_condition] * ctrl_adata.shape[0])

        net_ctrl_adata = AnnDatasetFromAnnData(ctrl_adata)

        posterior = self.trainer.create_posterior(self.trainer.model, net_ctrl_adata, indices=np.arange(len(net_ctrl_adata)))

        generated_samples, _ = posterior.sequential().generate(n_generated_samples)

        reconstructed = generated_samples.mean(axis=2)
        reconstructed_adata = sc.AnnData(X=reconstructed)
        reconstructed_adata.obs = ctrl_adata.obs.copy(deep=True)
        reconstructed_adata.obs[condition_key].replace(source_condition,
                                                       f'{cell_type_to_predict}_pred_{target_condition}',
                                                       inplace=True)
        reconstructed_adata.var_names = cell_type_adata.var_names

        pred_adata = reconstructed_adata[
            reconstructed_adata.obs[condition_key] == f'{cell_type_to_predict}_pred_{target_condition}']

        sc.pp.normalize_per_cell(pred_adata)
        sc.pp.log1p(pred_adata)
        return pred_adata

    def train(self, adata, condition_key, cell_type_key, n_epochs=300, patience=10, lr_reducer=7):
        le = LabelEncoder()
        adata.obs['labels'] = le.fit_transform(adata.obs[cell_type_key].values)
        adata.obs['batch_indices'] = le.fit_transform(adata.obs[condition_key].values)

        net_adata = AnnDatasetFromAnnData(adata)

        early_stopping_kwargs = {
            "early_stopping_metric": "elbo",
            "save_best_state_metric": "elbo",
            "patience": patience,
            "threshold": 0,
            "reduce_lr_on_plateau": True,
            "lr_patience": lr_reducer,
            "lr_factor": 0.1,
        }

        self.trainer = UnsupervisedTrainer(
            self.model,
            net_adata,
            train_size=0.8,
            use_cuda=True,
            frequency=1,
            early_stopping_kwargs=early_stopping_kwargs,
        )

        self.trainer.train(n_epochs=n_epochs, lr=0.001)

    def restore_model(self):
        self.trainer.model.load_state_dict(torch.load(os.path.join(self.model_path, "scVI.pt")))
        self.trainer.model.eval()

    def save_model(self):
        os.makedirs("./models/scVI/subsample/", exist_ok=True)
        torch.save(self.trainer.model.state_dict(), os.path.join(self.model_path, "scVI.pt"))
