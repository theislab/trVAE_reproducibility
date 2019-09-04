from random import shuffle
import sys

import numpy as np
import scanpy.api as sc
import tensorflow as tf
from data_reader import data_reader


def train_test_split(adata, train_frac=0.85):
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


# =============================== downloading training and validation files ====================================
data_name = sys.argv[1]

data_path = f"../data/{data_name}/{data_name}.h5ad"

adata = sc.read(data_path)

if adata.shape[0] > 2000:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    
data, validation = train_test_split(adata, 0.80)

import os

print(os.getcwd())
# =============================== data gathering ====================================
if data_name == "haber":
    # training cells
    t_in = ['TA', 'TA.Early', 'Endocrine', 'Enterocyte', 'Enterocyte.Progenitor', 'Goblet', 'Stem']
    # heldout cells
    t_out = ['Tuft']
    
    source_conditions = ["Control"]
    target_conditions = ['Hpoly.Day10']
    
    specific_cell_type = "Tuft"

    cell_type_key = "cell_label"
elif data_name == "species":
    t_in = ['rabbit', 'mouse', 'pig']
    t_out = ['rat']
    
    source_conditions = ["unst"]
    target_conditions = ['LPS6']
    
    specific_cell_type = "rat"
    cell_type_key = "species"
elif data_name == "kang":
    t_in = ['CD14 Mono', 'CD4 T', 'B', 'CD16 Mono', 'CD8 T', 'T', 'DC']
    t_out = ['NK']
    
    source_conditions = ["CTRL"]
    target_conditions = ['STIM']
    
    specific_cell_type = "NK"

    cell_type_key = "cell_type"
    

dr = data_reader(data, validation, {"ctrl": source_conditions, "stim": target_conditions}, t_in, t_out, cell_type_key)
train_real = dr.train_real_adata
train_real_stim = train_real[train_real.obs["condition"].isin(target_conditions)]
train_real_ctrl = train_real[train_real.obs["condition"].isin(source_conditions)]
train_real_stim = train_real_stim.X
ind_list = [i for i in range(train_real_stim.shape[0])]
shuffle(ind_list)
train_real_stim = train_real_stim[ind_list, :]
gex_size = train_real_stim.shape[1]
train_real_ctrl = train_real_ctrl.X
ind_list = [i for i in range(train_real_ctrl.shape[0])]
shuffle(ind_list)
train_real_ctrl = train_real_ctrl[ind_list, :]
eq = min(len(train_real_ctrl), len(train_real_stim))
stim_ind = np.random.choice(range(len(train_real_stim)), size=eq, replace=False)
ctrl_ind = np.random.choice(range(len(train_real_ctrl)), size=eq, replace=False)
##selecting equal size for both stimulated and control cells
train_real_ctrl = train_real_ctrl[ctrl_ind, :]
train_real_stim = train_real_stim[stim_ind, :]
# =============================== parameters ====================================
model_to_use = "../models/STGAN/stgan"
os.makedirs(model_to_use, exist_ok=True)
X_dim = gex_size
z_dim = 100
h_dim = 200
batch_size = 512
inflate_to_size = 100
lambda_l2 = .8
arch = {"noise_input_size": z_dim, "inflate_to_size": inflate_to_size,
        "epochs": 0, "bsize": batch_size, "disc_internal_size ": h_dim, "#disc_train": 1}
X_stim = tf.placeholder(tf.float32, shape=[None, X_dim], name="data_stim")
X_ctrl = tf.placeholder(tf.float32, shape=[None, X_dim], name="data_ctrl")
time_step = tf.placeholder(tf.int32)
size = tf.placeholder(tf.int32)
learning_rate = 0.001
initializer = tf.truncated_normal_initializer(stddev=0.02)
is_training = tf.placeholder(tf.bool)
dr_rate = .5
const = 5


### helper function


def predict(ctrl):
    pred = sess.run(gen_stim_fake, feed_dict={X_ctrl: ctrl, is_training: False})
    return pred


def low_embed(all):
    pred = sess.run(disc_c, feed_dict={X_ctrl: all, is_training: False})
    return pred


def low_embed_stim(all):
    pred = sess.run(disc_s, feed_dict={X_stim: all, is_training: False})
    return pred


# network

def discriminator_stimulated(tensor, reuse=False, ):
    with tf.variable_scope("discriminator_s", reuse=reuse):
        h = tf.layers.dense(inputs=tensor, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)
        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        disc = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(disc, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=1, kernel_initializer=initializer, use_bias=False)
        h = tf.nn.sigmoid(h)

        return h, disc


def discriminator_control(tensor, reuse=False, ):
    with tf.variable_scope("discriminator_b", reuse=reuse):
        h = tf.layers.dense(inputs=tensor, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        disc = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(disc, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=1, kernel_initializer=initializer, use_bias=False)
        h = tf.nn.sigmoid(h)
        return h, disc


def generator_stim_ctrl(image, reuse=False):
    with tf.variable_scope("generator_sb", reuse=reuse):
        h = tf.layers.dense(inputs=image, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=50, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.relu(h)
        return h


def generator_ctrl_stim(image, reuse=False, ):
    with tf.variable_scope("generator_bs", reuse=reuse):
        h = tf.layers.dense(inputs=image, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=50, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.relu(h)

        return h


# generator and discriminator

gen_stim_fake = generator_ctrl_stim(X_ctrl)
gen_ctrl_fake = generator_stim_ctrl(X_stim)

recon_ctrl = generator_stim_ctrl(gen_stim_fake, reuse=True)
recon_stim = generator_ctrl_stim(gen_ctrl_fake, reuse=True)

disc_ctrl_fake, _ = discriminator_control(gen_ctrl_fake)
disc_stim_fake, _ = discriminator_stimulated(gen_stim_fake)

disc_ctrl_real, disc_c = discriminator_control(X_ctrl, reuse=True)
disc_stim_real, disc_s = discriminator_stimulated(X_stim, reuse=True)

# computing loss

const_loss_s = tf.reduce_sum(tf.losses.mean_squared_error(recon_ctrl, X_ctrl))
const_loss_b = tf.reduce_sum(tf.losses.mean_squared_error(recon_stim, X_stim))

gen_ctrl_loss = tf.reduce_sum(tf.square(disc_ctrl_fake - 1)) / 2
gen_stim_loss = tf.reduce_sum(tf.square(disc_stim_fake - 1)) / 2

disc_ctrl_loss = tf.reduce_sum(tf.square(disc_ctrl_real - 1) + tf.square(disc_ctrl_fake)) / 2
disc_stim_loss = tf.reduce_sum(tf.square(disc_stim_real - 1) + tf.square(disc_stim_fake)) / 2

gen_loss = const * (const_loss_s + const_loss_b) + gen_ctrl_loss + gen_stim_loss
disc_loss = disc_ctrl_loss + disc_stim_loss

# applying gradients

gen_sb_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_sb")
gen_bs_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_bs")
dis_s_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_s")
dis_b_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_b")
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    update_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss,
                                                                            var_list=dis_s_variables + dis_b_variables,
                                                                            )
    update_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(gen_loss,
                                                                            var_list=gen_sb_variables + gen_bs_variables)
global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer().run()


def train(n_epochs, initial_run=True):
    if initial_run:
        print("Initial run")
        print("Training started")
        assign_step_zero = tf.assign(global_step, 0)
        init_step = sess.run(assign_step_zero)
    if not initial_run:
        saver.restore(sess, model_to_use)
        current_step = sess.run(global_step)
    for it in range(n_epochs):
        increment_global_step_op = tf.assign(global_step, global_step + 1)
        step = sess.run(increment_global_step_op)
        current_step = sess.run(global_step)
        batch_ind1 = np.random.choice(range(len(train_real_stim)), size=eq, replace=False)
        mb_ctrl = train_real_ctrl[batch_ind1, :]
        mb_stim = train_real_stim[batch_ind1, :]
        for gen_it in range(2):
            _, g_loss, d_loss = sess.run([update_G, gen_loss, disc_loss],
                                         feed_dict={X_ctrl: mb_ctrl, X_stim: mb_stim, is_training: True})
        _, g_loss, d_loss = sess.run([update_G, gen_loss, disc_loss],
                                     feed_dict={X_ctrl: mb_ctrl, X_stim: mb_stim, is_training: True})
        print(f"Iteration {it}: {g_loss + d_loss}")
        _ = sess.run(update_D, feed_dict={X_ctrl: mb_ctrl, X_stim: mb_stim, is_training: True})
    save_path = saver.save(sess, model_to_use)
    print("Model saved in file: %s" % save_path)
    print(f"Training finished")


def restore():
    saver.restore(sess, model_to_use)


if __name__ == "__main__":
    path_to_save = f"../results/CycleGAN/{data_name}/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = path_to_save
    sc.settings.writedir = "../data"
    if sys.argv[2] == "train":
        train(1000, initial_run=True)
    else:
        restore()
    print("model has been trained/restored!")
    adata_list = dr.extractor(data, specific_cell_type)
    ctrl_CD4T = adata_list[1]
    if sys.argv[2] == "train":
        predicted_cells = predict(ctrl_CD4T.X)
        all_Data = sc.AnnData(np.concatenate([adata_list[1].X, adata_list[2].X, predicted_cells]))
        all_Data.obs["condition"] = ["ctrl"] * len(adata_list[1].X) + [f"real_{target_conditions[0]}"] * len(adata_list[2].X) + \
                                    [f"pred_{target_conditions[0]}"] * len(predicted_cells)
        all_Data.var_names = adata_list[3].var_names
        all_Data.write(f"../data/reconstructed/{data_name}/cgan-{specific_cell_type}.h5ad")
    elif sys.argv[2] == "latent":
        low_dim = low_embed_stim(train_real.X)
        dt = sc.AnnData(low_dim)
        sc.pp.neighbors(dt)
        sc.tl.umap(dt)
        dt.obs["cell_label"] = train_real.obs["cell_label"]
        dt.obs["condition"] = train_real.obs["condition"]
        sc.pl.umap(dt, color=["cell_label"], show=False, frameon=False
                   , save="_latent_cell_label.png")

        sc.pl.umap(dt, color=["condition"], show=False, frameon=False
                   , save="_latent_condition.png", palette=["#96a1a3", "#A4E804"])

        os.rename(src=os.path.join(path_to_save, "umap_latent_cell_label.png"),
                  dst=os.path.join(path_to_save, f"SupplFig4b_style_transfer_celltype.png"))

        os.rename(src=os.path.join(path_to_save, "umap_latent_condition.png"),
                  dst=os.path.join(path_to_save, f"SupplFig4b_style_transfer_condition.png"))
