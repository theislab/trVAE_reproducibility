import math
from random import shuffle

import anndata
import numpy as np
import scanpy as sc
import tensorflow as tf
from scipy import sparse
from sklearn.preprocessing import LabelEncoder


def asinh(x, scale=5.):
    """Asinh transform."""
    f = np.vectorize(lambda y: math.asinh(y / scale))
    return f(x)


def sinh(x, scale=5.):
    """Reverse transform for asinh."""
    return scale * np.sinh(x)


def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky ReLU activation."""
    return tf.maximum(x, leak * x)


def tbn(name):
    """Get the tensor in the default graph of the given name."""
    return tf.get_default_graph().get_tensor_by_name(name)


def obn(name):
    """Get the operation node in the default graph of the given name."""
    return tf.get_default_graph().get_operation_by_name(name)


def calculate_mmd(k1, k2, k12):
    """ Calculates MMD given kernels for batch1, batch2, and between batches """
    return k1.sum() / (k1.shape[0] * k1.shape[1]) + k2.sum() / (k2.shape[0] * k2.shape[1]) - 2 * k12.sum() / (
            k12.shape[0] * k12.shape[1])


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True,
              n_top_genes=2000):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    adata_count = adata.copy()

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if n_top_genes > 0 and adata.shape[1] > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        genes = adata.var['highly_variable']
        adata = adata[:, genes]
        adata_count = adata_count[:, genes]

    if normalize_input:
        sc.pp.scale(adata)

    if sparse.issparse(adata_count.X):
        adata_count.X = adata_count.X.A

    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata_count.copy()
    else:
        adata.raw = adata_count

    return adata


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
        le = label_encoder
        labels = np.zeros(adata.shape[0])
        for condition, label in label_encoder.items():
            labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), le


def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    return adata


def create_dictionary(conditions, target_conditions=[]):
    if isinstance(target_conditions, list):
        target_conditions = [target_conditions]

    dictionary = {}
    conditions = [e for e in conditions if e not in target_conditions]
    for idx, condition in enumerate(conditions):
        dictionary[condition] = idx
    return dictionary

def data_remover(adata, remain_list, remove_list, cell_type_key, condition_key):
    """
        Removes specific cell type in stimulated condition form `adata`.

        # Parameters
            adata: `~anndata.AnnData`
                Annotated data matrix
            remain_list: list
                list of cell types which are going to be remained in `adata`.
            remove_list: list
                list of cell types which are going to be removed from `adata`.

        # Returns
            merged_data: list
                returns array of specified cell types in stimulated condition

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./data/train_kang.h5ad")
        remove_list = ["CD14+Mono", "CD8T"]
        remain_list = ["CD4T", "Dendritic"]
        filtered_data = data_remover(train_data, remain_list, remove_list)
        ```
    """
    source_data = []
    for i in remain_list:
        source_data.append(extractor(adata, i, conditions={"ctrl": "control", "stim": "stimulated"},
                                     cell_type_key=cell_type_key, condition_key=condition_key)[3])
    target_data = []
    for i in remove_list:
        target_data.append(extractor(adata, i, conditions={"ctrl": "control", "stim": "stimulated"},
                                     cell_type_key=cell_type_key, condition_key=condition_key)[1])
    merged_data = training_data_provider(source_data, target_data)
    merged_data.var_names = adata.var_names
    return merged_data


def extractor(data, cell_type, conditions, cell_type_key="cell_type", condition_key="condition"):
    """
        Returns a list of `data` files while filtering for a specific `cell_type`.

        # Parameters
        data: `~anndata.AnnData`
            Annotated data matrix
        cell_type: basestring
            specific cell type to be extracted from `data`.
        conditions: dict
            dictionary of stimulated/control of `data`.

        # Returns
            list of `data` files while filtering for a specific `cell_type`.

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./data/train.h5ad")
        test_data = anndata.read("./data/test.h5ad")
        train_data_extracted_list = extractor(train_data, "CD4T", conditions={"ctrl": "control", "stim": "stimulated"})
        ```

    """
    cell_with_both_condition = data[data.obs[cell_type_key] == cell_type]
    condtion_1 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["ctrl"])]
    condtion_2 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"])]
    training = data[~((data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"]))]
    return [training, condtion_1, condtion_2, cell_with_both_condition]


def training_data_provider(train_s, train_t):
    """
        Concatenates two lists containing adata files

        # Parameters
        train_s: `~anndata.AnnData`
            Annotated data matrix.
        train_t: `~anndata.AnnData`
            Annotated data matrix.

        # Returns
            Concatenated Annotated data matrix.

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./data/train_kang.h5ad")
        test_data = anndata.read("./data/test.h5ad")
        whole_data = training_data_provider(train_data, test_data)
        ```
    """
    train_s_X = []
    train_s_diet = []
    train_s_groups = []
    for i in train_s:
        train_s_X.append(i.X.A)
        train_s_diet.append(i.obs["condition"].tolist())
        train_s_groups.append(i.obs["cell_type"].tolist())
    train_s_X = np.concatenate(train_s_X)
    temp = []
    for i in train_s_diet:
        temp = temp + i
    train_s_diet = temp
    temp = []
    for i in train_s_groups:
        temp = temp + i
    train_s_groups = temp
    train_t_X = []
    train_t_diet = []
    train_t_groups = []
    for i in train_t:
        train_t_X.append(i.X.A)
        train_t_diet.append(i.obs["condition"].tolist())
        train_t_groups.append(i.obs["cell_type"].tolist())
    temp = []
    for i in train_t_diet:
        temp = temp + i
    train_t_diet = temp
    temp = []
    for i in train_t_groups:
        temp = temp + i
    train_t_groups = temp
    train_t_X = np.concatenate(train_t_X)
    train_real = np.concatenate([train_s_X, train_t_X])  # concat all
    train_real = anndata.AnnData(train_real)
    train_real.obs["condition"] = train_s_diet + train_t_diet
    train_real.obs["cell_type"] = train_s_groups + train_t_groups
    return train_real


def balancer(adata, cell_type_key="cell_type", condition_key="condition"):
    """
        Makes cell type population equal.

        # Parameters
        adata: `~anndata.AnnData`
            Annotated data matrix.

        # Returns
            balanced_data: `~anndata.AnnData`
                Equal cell type population Annotated data matrix.

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./train_kang.h5ad")
        train_ctrl = train_data[train_data.obs["condition"] == "control", :]
        train_ctrl = balancer(train_ctrl)
        ```
    """
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata.copy()[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    all_data_x = []
    all_data_label = []
    all_data_condition = []
    for cls in class_names:
        temp = adata.copy()[adata.obs[cell_type_key] == cls]
        index = np.random.choice(range(len(temp)), max_number)
        if sparse.issparse(temp.X):
            temp_x = temp.X.A[index]
        else:
            temp_x = temp.X[index]
        all_data_x.append(temp_x)
        temp_ct = np.repeat(cls, max_number)
        all_data_label.append(temp_ct)
        temp_cc = np.repeat(np.unique(temp.obs[condition_key]), max_number)
        all_data_condition.append(temp_cc)
    balanced_data = anndata.AnnData(np.concatenate(all_data_x))
    balanced_data.obs[cell_type_key] = np.concatenate(all_data_label)
    balanced_data.obs[condition_key] = np.concatenate(all_data_label)
    class_names = np.unique(balanced_data.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = len(balanced_data[balanced_data.obs[cell_type_key] == cls])
    return balanced_data


def shuffle_data(adata, labels=None):
    """
        Shuffles the `adata`.

        # Parameters
        adata: `~anndata.AnnData`
            Annotated data matrix.
        labels: numpy nd-array
            list of encoded labels

        # Returns
            adata: `~anndata.AnnData`
                Shuffled annotated data matrix.
            labels: numpy nd-array
                Array of shuffled labels if `labels` is not None.

        # Example
        ```python
        import scgen
        import anndata
        import pandas as pd
        train_data = anndata.read("./data/train.h5ad")
        train_labels = pd.read_csv("./data/train_labels.csv", header=None)
        train_data, train_labels = shuffle_data(train_data, train_labels)
        ```
    """
    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    if sparse.issparse(adata.X):
        x = adata.X.A[ind_list, :]
    else:
        x = adata.X[ind_list, :]
    if labels is not None:
        labels = labels[ind_list]
        adata = anndata.AnnData(x, obs={"labels": list(labels)})
        return adata, labels
    else:
        return anndata.AnnData(x, obs=adata.obs)
