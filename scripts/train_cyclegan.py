import sys

import scanpy.api as sc

import reptrvae
from reptrvae.models import CycleGAN
from reptrvae.utils import train_test_split

data_name = sys.argv[1]

if data_name == "haber":
    keys = ["Control", "Hpoly.Day10"]
    specific_cell_type = "Tuft"
    cell_type_key = "cell_label"
    condition_key = "condition"
    source_condition = "Control"
    target_condition = 'Hpoly.Day10'
    target_conditions = ['Hpoly.Day10']
    le = {"Control": 0, "Hpoly.Day10": 1}
elif data_name == "species":
    keys = ["unst", "LPS6"]
    specific_cell_type = "rat"
    cell_type_key = "species"
    condition_key = "condition"
    source_condition = "unst"
    target_condition = "LPS6"
    target_conditions = ['LPS6']
    le = {"unst": 0, "LPS6": 1}
elif data_name == "kang":
    keys = ["control", "stimulated"]
    specific_cell_type = "NK"
    cell_type_key = "cell_type"
    condition_key = "condition"
    source_condition = "control"
    target_condition = "stimulated"
    target_conditions = ['stimulated']
    le = {"control": 0, "stimulated": 1}
else:
    raise Exception("Invalid Data name")

adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

if adata.shape[1] > 2000:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
print(adata)
train_adata, valid_adata = train_test_split(adata, 0.80)
print(train_adata.shape, valid_adata.shape)
net_train_adata = train_adata[
    ~((train_adata.obs[cell_type_key] == specific_cell_type) & (
        train_adata.obs[condition_key].isin(target_conditions)))]
net_valid_adata = valid_adata[
    ~((valid_adata.obs[cell_type_key] == specific_cell_type) & (
        valid_adata.obs[condition_key].isin(target_conditions)))]
print(net_train_adata.shape, net_valid_adata.shape)
model = CycleGAN(train_adata.shape[1], z_dimension=40)

model.train(net_train_adata, net_valid_adata, condition_key=condition_key,
            source_condition=source_condition, target_condition=target_condition,
            n_epochs=1000, batch_size=64)

latent_adata = model.to_latent(net_train_adata)

pred_adata = model.predict(adata, specific_cell_type, source_condition, condition_key, cell_type_key)

pred_adata.obs[condition_key] = [f"{specific_cell_type}_pred_{target_condition}"] * pred_adata.shape[0]

pred_adata.write_h5ad(f"./data/reconstructed/{data_name}/CycleGAN-{specific_cell_type}.h5ad")

reptrvae.pl.plot_umap(latent_adata,
                      condition_key, cell_type_key,
                      frameon=False, path_to_save=f"./results/{data_name}/", model_name="CycleGAN")
