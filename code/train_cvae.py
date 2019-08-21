import numpy as np
import rcvae
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

data_name = "haber"
specific_cell_type = "Tuft"
cell_type_key = "cell_label"
condition_key = "condition"
control_condition = "Control"
target_condition = 'Hpoly.Day10'


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

keys = ["Control", "Hpoly.Day10"]

adata = sc.read(f"../data/{data_name}/{data_name}.h5ad")
adata = adata.copy()[adata.obs[condition_key].isin(keys)]

train_adata, valid_adata = train_test_split(adata, 0.80)

le = {"Control": 0, "Hpoly.Day10": 1}

target_conditions = ['Hpoly.Day10']

net_train_adata = train_adata[
    ~((train_adata.obs[cell_type_key] == specific_cell_type) & (
        train_adata.obs[condition_key].isin(target_conditions)))]
net_valid_adata = valid_adata[
    ~((valid_adata.obs[cell_type_key] == specific_cell_type) & (
        valid_adata.obs[condition_key].isin(target_conditions)))]

z_dim = 20
network = rcvae.CVAE(x_dimension=net_train_adata.X.shape[1],
                     z_dimension=z_dim,
                     alpha=0.1,
                     model_path="../models/CVAE/haber/Tuft/cvae")
network.train(net_train_adata,
              use_validation=True,
              valid_data=net_valid_adata,
              n_epochs=100)

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

all_adata.write(f"../data/reconstructed/CVAE_{specific_cell_type}.h5ad")
