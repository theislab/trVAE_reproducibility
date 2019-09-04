import anndata
import numpy as np
import scanpy as sc
import scgen
from scipy import sparse


def train_test_split(adata, train_frac=0.85):
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


def test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=50,
                                           alpha=0.1,
                                           n_epochs=1000,
                                           batch_size=32,
                                           dropout_rate=0.25,
                                           learning_rate=0.001,
                                           condition_key="condition",
                                           cell_type_to_train=None):
    if data_name == "haber":
        keys = ['Control', 'Hpoly.Day10']
        stim_keys = ["Hpoly.Day10"]
        cell_type_key = "cell_label"
    elif data_name == "species":
        keys = ['unst', 'LPS6']
        stim_keys = ["LPS6"]
        cell_type_key = "species"
    elif data_name == "kang":
        keys = ['CTRL', 'STIM']
        stim_keys = ["STIM"]
        cell_type_key = "cell_type"

    adata = sc.read(f"../data/{data_name}/{data_name}.h5ad")
    adata = adata.copy()[adata.obs[condition_key].isin(keys)]

    train_adata, valid_adata = train_test_split(adata, 0.80)

    for cell_type in train_adata.obs[cell_type_key].unique().tolist():
        if cell_type_to_train is not None and cell_type != cell_type_to_train:
            continue
        print(f"Training for {cell_type}")
        net_train_adata = train_adata[~((train_adata.obs[cell_type_key] == cell_type) & (train_adata.obs[condition_key].isin(stim_keys)))]
        net_valid_adata = valid_adata[~((valid_adata.obs[cell_type_key] == cell_type) & (valid_adata.obs[condition_key].isin(stim_keys)))]
        network = scgen.VAEArith(x_dimension=net_train_adata.X.shape[1],
                                 z_dimension=z_dim,
                                 alpha=alpha,
                                 dropout_rate=dropout_rate,
                                 learning_rate=learning_rate,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")

        network.train(net_train_adata, use_validation=True, valid_data=net_valid_adata, n_epochs=n_epochs,
                      batch_size=batch_size,
                      verbose=True, early_stop_limit=5)
        network.sess.close()
        print(f"network_{cell_type} has been trained!")


def reconstruct_whole_data(data_name="pbmc", condition_key="condition", cell_type_to_predict=None):
    if data_name == "haber":
        keys = ["Control", "Hpoly.Day10"]
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
    elif data_name == "species":
        keys = ['unst', 'LPS6']
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
    elif data_name == "kang":
        keys = ['CTRL', 'STIM']
        stim_key = "STIM"
        ctrl_key = "CTRL"
        cell_type_key = "cell_type"

    adata = sc.read(f"../data/{data_name}/{data_name}.h5ad")
    adata = adata.copy()[adata.obs[condition_key].isin(keys)]

    all_data = anndata.AnnData()
    for idx, cell_type in enumerate(adata.obs[cell_type_key].unique().tolist()):
        if cell_type_to_predict is not None and cell_type != cell_type_to_predict:
            continue
        print(f"Reconstructing for {cell_type}")
        network = scgen.VAEArith(x_dimension=adata.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")
        network.restore_model()

        cell_type_data = adata[adata.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = adata[((adata.obs[cell_type_key] == cell_type) & (adata.obs[condition_key] == ctrl_key))]
        net_train_data = adata[~((adata.obs[cell_type_key] == cell_type) & (adata.obs[condition_key] == stim_key))]
        pred, delta = network.predict(adata=net_train_data,
                                      conditions={"ctrl": ctrl_key, "stim": stim_key},
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type)

        pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_{stim_key}"] * len(pred),
                                                cell_type_key: [cell_type] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                     obs={condition_key: [f"{cell_type}_ctrl"] * len(cell_type_ctrl_data),
                                          cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                     var={"var_names": cell_type_ctrl_data.var_names})
        if sparse.issparse(cell_type_data.X):
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.A
        else:
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                          obs={condition_key: [f"{cell_type}_real_{stim_key}"] * len(real_stim),
                                               cell_type_key: [cell_type] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        if idx == 0 or cell_type_to_predict is not None:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        print(f"Finish Reconstructing for {cell_type}")
        network.sess.close()
    all_data.write_h5ad(f"../data/reconstructed/{data_name}/scGen-{cell_type_to_predict}.h5ad")


if __name__ == '__main__':
    import sys
    data_name = sys.argv[1]

    if data_name == "haber":
        specific_cell_type = "Tuft"
    elif data_name == "kang":
        specific_cell_type = "NK"
    elif data_name == "species":
        specific_cell_type = "rat"

    test_train_whole_data_one_celltype_out(data_name, z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                                           dropout_rate=0.2, learning_rate=0.001, cell_type_to_train="rat")

    reconstruct_whole_data(data_name, cell_type_to_predict=specific_cell_type)
