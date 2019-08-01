# from hf import *

import numpy as np
import scanpy.api as sc
import scgen


# =============================== downloading training and validation files ====================================
# we do not use the validation data to apply vectroe arithmetics in gene expression space

def train(data_name="pbmc", cell_type="CD4T", p_type="unbiased"):
    train_path = f"../data/haber/train_{data_name}.h5ad"
    ctrl_key = "Control"
    stim_key = "Hpoly.Day10"
    stim_keys = ["Hpoly.Day3", "Hpoly.Day10", 'Salmonella']
    cell_type_key = "cell_label"
    data = sc.read(train_path)
    print("data has been loaded!")
    ctrl_cell = data[(data.obs["condition"] == ctrl_key) & (data.obs[cell_type_key] == cell_type)]
    stim_cell = data[(data.obs["condition"] == stim_key) & (data.obs[cell_type_key] == cell_type)]

    train_real_cd = data[data.obs["condition"] == ctrl_key, :]
    if p_type == "unbiased":
        train_real_cd = scgen.util.balancer(train_real_cd, cell_type_key=cell_type_key)
    train_real_stimulated = data[data.obs["condition"] == stim_key, :]
    train_real_stimulated = train_real_stimulated[train_real_stimulated.obs[cell_type_key] != cell_type]
    if p_type == "unbiased":
        train_real_stimulated = scgen.util.balancer(train_real_stimulated, cell_type_key=cell_type_key)

    import scipy.sparse as sparse
    if sparse.issparse(train_real_cd.X):
        train_real_cd = train_real_cd.X.A
        train_real_stimulated = train_real_stimulated.X.A
    else:
        train_real_cd = train_real_cd.X
        train_real_stimulated = train_real_stimulated.X
    if sparse.issparse(ctrl_cell.X):
        ctrl_cell.X = ctrl_cell.X.A
        stim_cell.X = stim_cell.X.A
    predicted_cells = predict(train_real_cd, train_real_stimulated, ctrl_cell.X)

    print("Prediction has been finished")
    all_Data = sc.AnnData(np.concatenate([ctrl_cell.X, stim_cell.X, predicted_cells]))
    all_Data.obs["condition"] = ["ctrl"] * ctrl_cell.shape[0] + ["real_stim"] * stim_cell.shape[0] + \
                                [f"{cell_type}_pred_{stim_key}"] * len(predicted_cells)
    all_Data.var_names = ctrl_cell.var_names
    if p_type == "unbiased":
        sc.write(f"../data/reconstructed/VecArithm_{cell_type}.h5ad", all_Data)
    else:
        sc.write(f"../data/reconstructed/VecArithm_{cell_type}_biased.h5ad", all_Data)


def predict(cd_x, hfd_x, cd_y, p_type="unbiased"):
    if p_type == "biased":
        cd_ind = np.arange(0, len(cd_x))
        stim_ind = np.arange(0, len(hfd_x))
    else:
        eq = min(len(cd_x), len(hfd_x))
        cd_ind = np.random.choice(range(len(cd_x)), size=eq, replace=False)
        stim_ind = np.random.choice(range(len(hfd_x)), size=eq, replace=False)
    cd = np.average(cd_x[cd_ind, :], axis=0)
    stim = np.average(hfd_x[stim_ind, :], axis=0)
    delta = stim - cd
    predicted_cells = delta + cd_y
    return predicted_cells


if __name__ == "__main__":
    train("haber", "Tuft", "unbiased")