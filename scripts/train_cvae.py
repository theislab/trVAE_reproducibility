import sys

import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

from reptrvae.models import CVAE

data_name = sys.argv[1]


def train_test_split(adata, train_frac=0.85):
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


def label_encoder(adata, label_encoder=None, condition_key='condition'):
    if label_encoder is None:
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs[condition_key].tolist())
    else:
        le = None
        labels = np.zeros(adata.shape[0])
        for condition, label in label_encoder.items():
            labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), le


if data_name == "haber":
    keys = ["Control", "Hpoly.Day10"]
    specific_cell_type = "Tuft"
    cell_type_key = "cell_label"
    condition_key = "condition"
    control_condition = "Control"
    target_condition = 'Hpoly.Day10'
    target_conditions = ['Hpoly.Day10']
    le = {"Control": 0, "Hpoly.Day10": 1}

elif data_name == "species":
    keys = ["unst", "LPS6"]
    specific_cell_type = "rat"
    cell_type_key = "species"
    condition_key = "condition"
    control_condition = "unst"
    target_condition = "LPS6"
    target_conditions = ['LPS6']
    le = {"unst": 0, "LPS6": 1}
elif data_name == "kang":
    keys = ["CTRL", "STIM"]
    specific_cell_type = "NK"
    cell_type_key = "cell_type"
    condition_key = "condition"
    control_condition = "CTRL"
    target_condition = "STIM"
    target_conditions = ['STIM']
    le = {"CTRL": 0, "STIM": 1}

adata = sc.read(f"../data/{data_name}/{data_name}_normalized.h5ad")
adata = adata.copy()[adata.obs[condition_key].isin(keys)]

if adata.shape[1] > 2000:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]

train_adata, valid_adata = train_test_split(adata, 0.80)

net_train_adata = train_adata[
    ~((train_adata.obs[cell_type_key] == specific_cell_type) & (
        train_adata.obs[condition_key].isin(target_conditions)))]
net_valid_adata = valid_adata[
    ~((valid_adata.obs[cell_type_key] == specific_cell_type) & (
        valid_adata.obs[condition_key].isin(target_conditions)))]

z_dim = 20
network = CVAE(x_dimension=net_train_adata.X.shape[1],
               z_dimension=z_dim,
               alpha=0.1,
               model_path=f"../models/CVAE/{data_name}/{specific_cell_type}/cvae")
if sys.argv[2] == 'scripts':
    network.train(net_train_adata,
                  use_validation=True,
                  valid_data=net_valid_adata,
                  n_epochs=300)

    train_labels, _ = label_encoder(train_adata, le, 'condition')
    cell_type_adata = train_adata[train_adata.obs[cell_type_key] == specific_cell_type]

    unperturbed_data = cell_type_adata[cell_type_adata.obs[condition_key] == control_condition]
    target_labels = np.zeros((len(unperturbed_data), 1)) + le[target_condition]
    predicted_cells = network.predict(unperturbed_data, target_labels)
    pred_adata = sc.AnnData(predicted_cells,
                            obs={condition_key: [f"{specific_cell_type}_pred_{target_condition}"] * len(target_labels)})
    pred_adata.obs['method'] = 'CVAE'
    pred_adata.obs[cell_type_key] = specific_cell_type
    pred_adata.var_names = cell_type_adata.var_names
    all_adata = cell_type_adata.concatenate(pred_adata)

    all_adata.write(f"../data/reconstructed/{data_name}/CVAE-{specific_cell_type}.h5ad")
    print("Model has been trained")
    sc.settings.figdir = "../results/kang/"
    labels, _ = label_encoder(adata, condition_key=condition_key)

    latent = network.to_latent(adata.X, labels)
    latent_adata = sc.AnnData(X=latent)
    latent_adata.obs = adata.obs.copy(deep=True)

    mmd_latent = network.to_mmd_layer(adata.X, labels)
    mmd_latent_adata = sc.AnnData(X=mmd_latent)
    mmd_latent_adata.obs = adata.obs.copy(deep=True)

    print("Latents has been computed")

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=condition_key, frameon=False,
               save='_CVAE_latent_condition.pdf')
    sc.pl.umap(latent_adata, color=cell_type_key, frameon=False,
               save='_CVAE_latent_celltype.pdf')

    sc.pp.neighbors(mmd_latent_adata)
    sc.tl.umap(mmd_latent_adata)
    sc.pl.umap(mmd_latent_adata, color=condition_key, frameon=False,
               save='_CVAE_MMD_latent_condition.pdf')
    sc.pl.umap(mmd_latent_adata, color=cell_type_key, frameon=False,
               save='_CVAE_MMD_latent_celltypes.pdf')
    print("Latents has been plotted and saved")

else:
    sc.settings.figdir = "../results/kang/"
    network.restore_model()
    labels, _ = label_encoder(adata, condition_key=condition_key)

    latent = network.to_latent(adata.X, labels)
    latent_adata = sc.AnnData(X=latent)
    latent_adata.obs = adata.obs.copy(deep=True)

    mmd_latent = network.to_mmd_layer(adata.X, labels)
    mmd_latent_adata = sc.AnnData(X=mmd_latent)
    mmd_latent_adata.obs = adata.obs.copy(deep=True)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=condition_key, frameon=False,
               save='_CVAE_latent_condition.pdf')
    sc.pl.umap(latent_adata, color=cell_type_key, frameon=False,
               save='_CVAE_latent_celltype.pdf')

    sc.pp.neighbors(mmd_latent_adata)
    sc.tl.umap(mmd_latent_adata)
    sc.pl.umap(mmd_latent_adata, color=condition_key, frameon=False,
               save='_CVAE_MMD_latent_condition.pdf')
    sc.pl.umap(mmd_latent_adata, color=cell_type_key, frameon=False,
               save='_CVAE_MMD_latent_celltypes.pdf')
