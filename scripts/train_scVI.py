import sys

import numpy as np
import scanpy as sc

import reptrvae
from reptrvae.models._scvi import scVI

data_name = sys.argv[1]

if data_name == "haber":
    cell_type_key = 'cell_label'
    condition_key = 'condition'
    specific_celltype = "Tuft"
    source_conditions = ["Control"]
    target_conditions = ["Hpoly.Day10"]
    conditions = ['Control', 'Hpoly.Day10']
elif data_name == "kang":
    cell_type_key = 'cell_type'
    condition_key = 'condition'
    specific_celltype = "NK"
    source_conditions = ["CTRL"]
    target_conditions = ["STIM"]
    conditions = ['CTRL', 'STIM']
else:
    raise Exception("Invalid data_name")

adata = sc.read(f"./data/{data_name}/{data_name}_count.h5ad")
adata = adata.copy()[adata.obs[condition_key].isin(conditions)]

adata_normalized = adata.copy()
sc.pp.normalize_per_cell(adata_normalized)
sc.pp.log1p(adata_normalized)

sc.pp.highly_variable_genes(adata_normalized, n_top_genes=2000)

adata = adata[:, adata_normalized.var['highly_variable']]

adata.X = np.array(adata.X, dtype='int32')

net_adata = adata[
    ~((adata.obs[cell_type_key] == specific_celltype) & (adata.obs[condition_key].isin(target_conditions)))]

n_batches = len(net_adata.obs[condition_key].unique().tolist())

model = scVI(x_dimension=net_adata.shape[1], n_batches=n_batches, model_path="./models/")

model.train(net_adata, condition_key=condition_key, cell_type_key=cell_type_key, n_epochs=1000, patience=50,
            lr_reducer=40)

model.save_model()

latent_adata = model.to_latent(net_adata, condition_key=condition_key, cell_type_key=cell_type_key)

pred_adata = model.predict(net_adata, cell_type_to_predict=specific_celltype, condition_key=condition_key,
                           cell_type_key=cell_type_key,
                           target_condition=target_conditions[0], source_condition=source_conditions[0],
                           n_generated_samples=50)

reptrvae.pl.plot_umap(latent_adata, condition_key=condition_key, cell_type_key=cell_type_key, frameon=False,
                      path_to_save=f"./results/{data_name}/", model_name="scVI_latent")

pred_adata.write_h5ad(f"./data/reconstructed/{data_name}/scVI-{specific_celltype}.h5ad")
