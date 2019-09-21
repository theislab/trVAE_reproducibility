import sys
import os
import numpy as np
import scanpy as sc

import reptrvae
from reptrvae.utils import remove_sparsity

data_name = sys.argv[1]

if data_name == "celeba":
    conditions = ["-1", "1"]
    target_conditions = ["-1"]
    source_condition = "-1"
    target_condition = "1"
    labelencoder = {"-1": 0, "1": 1}
    label_key = "labels"
    condition_key = "condition"
    specific_labels = ["1"]
    arch_style = 3
    adata = reptrvae.dl.prepare_and_load_celeba(file_path="./data/celeba/img_align_celeba.zip",
                                                attr_path="./data/celeba/list_attr_celeba.txt",
                                                landmark_path="../data/celeba/list_landmarks_align_celeba.txt",
                                                gender="Male",
                                                attribute="Smiling",
                                                max_n_images=50000,
                                                img_width=64,
                                                img_height=64,
                                                restore=True,
                                                save=True)
    input_shape = (64, 64, 3)
elif data_name == "mnist":
    conditions = ["normal", "thin", "thick"]
    target_conditions = ["thin", 'thick']
    source_condition = "normal"
    labelencoder = {"normal": 0, "thin": 1, "thick": 2}
    label_key = "labels"
    condition_key = "condition"
    specific_labels = [1, 3, 6, 7]
    arch_style = 1
    adata = sc.read("./data/thick_thin_mnist/thick_thin_mnist.h5ad")
    input_shape = (28, 28, 1)
else:
    raise Exception("Invalid data name")

adata = remove_sparsity(adata)

train_adata, valid_adata = reptrvae.utils.train_test_split(adata, 0.80)

net_train_adata = train_adata[
    ~((train_adata.obs[label_key].isin(specific_labels)) & (train_adata.obs[condition_key].isin(target_conditions)))]
net_valid_adata = valid_adata[
    ~((valid_adata.obs[label_key].isin(specific_labels)) & (valid_adata.obs[condition_key].isin(target_conditions)))]

network = reptrvae.models.DCtrVAE(x_dimension=input_shape,
                                  z_dimension=60,
                                  n_conditions=len(net_train_adata.obs[condition_key].unique()),
                                  alpha=5e-5,
                                  beta=500,
                                  eta=100,
                                  clip_value=1e6,
                                  lambda_l1=0.0,
                                  lambda_l2=0.0,
                                  learning_rate=0.001,
                                  model_path=f"./models/trVAE/best/{data_name}-{specific_labels}/",
                                  dropout_rate=0.2,
                                  output_activation='relu')

network.train(net_train_adata,
              net_valid_adata,
              labelencoder,
              condition_key,
              n_epochs=10000,
              batch_size=1024,
              verbose=2,
              early_stop_limit=150,
              lr_reducer=100,
              shuffle=True,
              )

train_labels, _ = reptrvae.tl.label_encoder(net_train_adata, labelencoder, condition_key)
# mmd_adata = network.to_mmd_layer(net_train_adata, train_labels, feed_fake=-1)

pred_adata_all = None
for target_condition in target_conditions:
    for specific_label in specific_labels:
        label_adata = adata[adata.obs[label_key] == specific_label]
        source_adata = label_adata[label_adata.obs[condition_key] == source_condition]
        source_labels = np.zeros(source_adata.shape[0]) + labelencoder[source_condition]
        target_labels = np.zeros(source_adata.shape[0]) + labelencoder[target_condition]

        pred_adata = network.predict(source_adata,
                                     encoder_labels=source_labels,
                                     decoder_labels=target_labels,
                                     )

        pred_adata.obs[condition_key] = [f"{specific_label}_pred_{target_condition}"] * pred_adata.shape[0]

        pred_adata_all = pred_adata.copy() if pred_adata_all is None else pred_adata_all.concatenate(pred_adata)

os.makedirs(f"./data/reconstructed/{data_name}/", exist_ok=True)
pred_adata_all.write_h5ad(f"./data/reconstructed/{data_name}/trVAE.h5ad")

os.makedirs(f"./results/{data_name}/", exist_ok=True)
# reptrvae.pl.plot_umap(mmd_adata,
#                       condition_key, cell_type_key,
#                       frameon=False, path_to_save=f"./results/{data_name}/", model_name="trVAE_MMD",
#                       ext="png")
