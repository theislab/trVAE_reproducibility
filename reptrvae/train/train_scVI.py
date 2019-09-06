import sys
import scanpy as sc
import numpy as np
from scvi.dataset import AnnDatasetFromAnnData
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
from sklearn.preprocessing import LabelEncoder

data_name = sys.argv[1]

if data_name == "haber":
    cell_type_key = 'cell_label'
    condition_key = 'condition'
    specific_celltype = "Tuft"
    source_conditions = ["Control"]
    target_conditions = ["Hpoly.Day10"]
    conditions = ['Control', 'Hpoly.Day10']
    target_condition = "Hpoly.Day10"
elif data_name == "species":
    cell_type_key = 'species'
    condition_key = 'condition'
    specific_celltype = "rat"
    source_conditions = ["unst"]
    target_conditions = ["LPS6"]
    conditions = ['unst', 'LPS6']
    target_condition = "LPS6"
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

net_adata = adata[~((adata.obs[cell_type_key] == specific_celltype) & (adata.obs[condition_key].isin(target_conditions)))]
cell_type_adata = adata[adata.obs[cell_type_key] == specific_celltype]
ctrl_adata = cell_type_adata[cell_type_adata.obs[condition_key].isin(source_conditions)]
real_adata = cell_type_adata[cell_type_adata.obs[condition_key].isin(target_conditions)]

le = LabelEncoder()
net_adata.obs['labels'] = le.fit_transform(net_adata.obs[cell_type_key].values)
net_adata.obs['batch_indices'] = le.fit_transform(net_adata.obs[condition_key].values)

net_adata = AnnDatasetFromAnnData(net_adata)


early_stopping_kwargs = {
        "early_stopping_metric": "elbo",
        "save_best_state_metric": "elbo",
        "patience": 50,
        "threshold": 0,
        "reduce_lr_on_plateau": True,
        "lr_patience": 30,
        "lr_factor": 0.1,
    }
use_batches = True

vae = VAE(net_adata.nb_genes, reconstruction_loss='nb', n_batch=net_adata.n_batches * use_batches)
trainer = UnsupervisedTrainer(
    vae,
    net_adata,
    train_size=0.8,
    use_cuda=True,
)

trainer.train(n_epochs=1000, lr=0.001)

real_adata.obs['batch_indices'] = le.transform(real_adata.obs[condition_key].values)
ctrl_adata.obs['batch_indices'] = le.transform([target_condition] * ctrl_adata.shape[0])

n_ctrl_samples = ctrl_adata.shape[0]

ctrl_adata = AnnDatasetFromAnnData(ctrl_adata)

full = trainer.create_posterior(trainer.model, ctrl_adata, indices=np.arange(len(ctrl_adata)))

generated_samples = full.sequential().generate(50)
generated_samples = generated_samples[0]

reconstructed = generated_samples.mean(axis=2)
reconstructed_adata = sc.AnnData(X=reconstructed)
reconstructed_adata.obs = ctrl_adata.obs
reconstructed_adata.obs[condition_key].replace(source_conditions[0], f'{target_condition}_pred', inplace=True)
reconstructed_adata.var_names = cell_type_adata.var_names

pred_adata = reconstructed_adata[reconstructed_adata.obs[condition_key] == f'{target_condition}_pred']

pred_adata.obs[condition_key] = pred_adata.obs[condition_key].replace(f"{target_condition}_pred", f"{specific_celltype}_pred_{target_condition}")
sc.pp.normalize_per_cell(pred_adata)
sc.pp.log1p(pred_adata)
pred_adata.write_h5ad(f"../trVAE_reproducibility/data/reconstructed/{data_name}/scVI-{specific_celltype}.h5ad")