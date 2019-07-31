import rcvae
import scanpy as sc
import numpy as np

def label_encoder(adata, label_encoder=None, condition_key='condition'):
    if label_encoder is None:
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(adata.obs[condition_key].tolist())
    else:
        le = None
        labels = np.zeros(adata.shape[0])
        for condition, label in label_encoder.items():
            labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), le
train = sc.read("../data/haber/train_haber.h5ad")
valid = sc.read("../data/haber/valid_haber.h5ad")
le = {"Control": 0, "Hpoly.Day3": 1, "Hpoly.Day10": 2, "Salmonella": 3}
target_conditions = ['Hpoly.Day3', 'Hpoly.Day10', 'Salmonella']
train = train[~((train.obs["cell_label"] == "Tuft") & (train.obs["condition"].isin(target_conditions)))]
valid = valid[~((valid.obs["cell_label"] == "Tuft") & (valid.obs["condition"].isin(target_conditions)))]
z_dim = 20
network = rcvae.CVAE(x_dimension=train.X.shape[1], z_dimension=z_dim, alpha=0.1, model_path="../models/CVAE/haber/Tuft/cvae")
network.train(train, use_validation=True, valid_data=valid, n_epochs=100)
labels, _ = label_encoder(train, le, 'condition')
train = sc.read("../data/haber/train_haber.h5ad")
CD4T = train[train.obs["cell_label"] == "Tuft"]
unperturbed_data = train[((train.obs["cell_label"] == "Tuft") & (train.obs["condition"] == "Control"))]
fake_labels = np.ones((len(unperturbed_data), 1)) + 1
predicted_cells = network.predict(unperturbed_data, fake_labels)
adata = sc.AnnData(predicted_cells, obs={"condition": ["pred_Hpoly.Day10"]*len(fake_labels)})
adata.var_names = CD4T.var_names
all_adata = CD4T.concatenate(adata)

all_adata.write("../data/reconstructed/CVAE_Tuft.h5ad")