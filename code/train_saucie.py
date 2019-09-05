import sys

import numpy as np
import scanpy as sc
from loader import Loader
from model import SAUCIE


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

adata = sc.read(f"../RCVAE/data/{data_name}/{data_name}_normalized.h5ad")
adata = adata.copy()[adata.obs[condition_key].isin(conditions)]

if adata.shape[1] > 2000:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]

train_adata, valid_adata = train_test_split(adata, 0.80)
cell_type_adata = adata[adata.obs[cell_type_key] == specific_celltype]
net_train_adata = train_adata[
    ~((train_adata.obs[cell_type_key] == specific_celltype) & (train_adata.obs[condition_key].isin(target_conditions)))]

real_adata = cell_type_adata[cell_type_adata.obs[condition_key] == target_condition]
ctrl_adata = cell_type_adata[cell_type_adata.obs[condition_key] == source_condition]

x_train = net_train_adata.X
y_train = label_encoder(net_train_adata, labelencoder, condition_key)
x_test = ctrl_adata.X
y_test = np.zeros(ctrl_adata.shape[0]) + labelencoder[target_condition]

train_loader = Loader(x_train, labels=y_train, shuffle=True)
real_loader = Loader(x_test, labels=y_test, shuffle=False)

saucie = SAUCIE(x_train.shape[1], lambda_c=.2, lambda_d=.4)

saucie.train(train_loader, 1000)

pred = saucie.get_reconstruction(real_loader)

pred_adata = sc.AnnData(X=pred[0])
pred_adata.obs[condition_key] = f"{specific_celltype}_pred_{target_condition}"
pred_adata.write_h5ad(f"../trVAE_reproducibility/data/reconstructed/{data_name}/SAUCIE-{specific_celltype}.h5ad")
