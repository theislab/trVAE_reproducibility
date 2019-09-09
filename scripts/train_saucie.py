import sys

import numpy as np
import scanpy as sc

import reptrvae
from reptrvae.models._saucie import SAUCIE


def label_encoder(adata, label_encoder=None, condition_key='condition'):
    labels = np.zeros(adata.shape[0])
    for condition, label in label_encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, )


def train_test_split(adata, train_frac=0.85):
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


data_name = sys.argv[1]

if data_name == "haber":
    conditions = ["Control", "Hpoly.Day10"]
    target_conditions = ["Hpoly.Day10"]
    source_condition = "Control"
    target_condition = "Hpoly.Day10"
    labelencoder = {"Control": 0, "Hpoly.Day10": 1}
    cell_type_key = "cell_label"
    condition_key = "condition"
    specific_celltype = "Tuft"

elif data_name == "kang":
    conditions = ["CTRL", "STIM"]
    target_conditions = ["STIM"]
    source_condition = "CTRL"
    target_condition = "STIM"
    labelencoder = {"CTRL": 0, "STIM": 1}
    cell_type_key = "cell_type"
    condition_key = "condition"
    specific_celltype = "NK"
else:
    raise Exception("InValid data name")

adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")
adata = adata.copy()[adata.obs[condition_key].isin(conditions)]

if adata.shape[1] > 2000:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]

train_adata, valid_adata = train_test_split(adata, 0.80)
cell_type_adata = adata[adata.obs[cell_type_key] == specific_celltype]
net_train_adata = train_adata[
    ~((train_adata.obs[cell_type_key] == specific_celltype) & (train_adata.obs[condition_key].isin(target_conditions)))]

model = SAUCIE(x_dimension=net_train_adata.shape[1], lambda_b=0.2, lambda_c=0.0, layer_c=0, lambda_d=0.0)

model.train(net_train_adata, condition_key=condition_key, le=labelencoder, n_epochs=1000, batch_size=256)
labels = np.ones_like(adata.obs[condition_key].values)
mmd_adata = model.to_latent(net_train_adata, labels)

pred_adata = model.predict(net_train_adata, labelencoder[target_condition], condition_key, cell_type_key,
                           specific_celltype,
                           source_condition, target_condition)

pred_adata.write_h5ad(f"./data/reconstructed/{data_name}/SAUCIE-{specific_celltype}.h5ad")

reptrvae.pl.plot_umap(mmd_adata,
                      condition_key, cell_type_key,
                      frameon=False, path_to_save=f"./results/{data_name}/", model_name="SAUCIE")