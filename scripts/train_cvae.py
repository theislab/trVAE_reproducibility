import sys

import numpy as np
import scanpy as sc

from reptrvae.models import CVAE
from reptrvae.plotting import plot_umap
from reptrvae.utils import train_test_split, label_encoder

data_name = sys.argv[1]

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
else:
    raise Exception("Invalid data name")

adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")
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

z_dim = 40
network = CVAE(x_dimension=net_train_adata.X.shape[1],
               z_dimension=z_dim,
               alpha=0.1,
               model_path=f"../models/CVAE/{data_name}/{specific_cell_type}/cvae")

network.train(net_train_adata,
              use_validation=True,
              valid_data=net_valid_adata,
              n_epochs=300)

train_labels, _ = label_encoder(train_adata, le, 'condition')
cell_type_adata = train_adata[train_adata.obs[cell_type_key] == specific_cell_type]

unperturbed_data = cell_type_adata[cell_type_adata.obs[condition_key] == control_condition]
target_labels = np.zeros((len(unperturbed_data), 1)) + le[target_condition]
pred_adata = network.predict(unperturbed_data, target_labels)
pred_adata.obs[condition_key] = [f"{specific_cell_type}_pred_{target_condition}"] * len(target_labels)
pred_adata.obs['method'] = 'CVAE'
pred_adata.write(f"../data/reconstructed/{data_name}/CVAE-{specific_cell_type}.h5ad")
print("Model has been trained")
sc.settings.figdir = f"../results/{data_name}/"
labels, _ = label_encoder(adata, condition_key=condition_key)
labels = np.ones_like(labels)
latent_adata = network.to_latent(adata, labels)
mmd_latent_adata = network.to_mmd_layer(adata, labels)

print("Latents has been computed")
plot_umap(mmd_latent_adata, condition_key, cell_type_key, False,
          path_to_save=f"./results/{data_name}/", model_name="CVAE_MMD")
print("Latents has been plotted and saved")
