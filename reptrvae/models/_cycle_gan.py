import logging
import os

import anndata
import keras
import numpy as np
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, ReLU
from keras.models import Model, load_model
from keras.optimizers import Adam

from reptrvae.models._network import Network
from reptrvae.utils import remove_sparsity

log = logging.getLogger(__file__)


class CycleGAN(Network):
    """
        Regularized C-VAE vector Network class. This class contains the implementation of Conditional
        Variational Auto-encoder network.
        # Parameters
            kwargs:
                key: `dropout_rate`: float
                        dropout rate
                key: `learning_rate`: float
                    learning rate of optimization algorithm
                key: `model_path`: basestring
                    path to save the model after training
                key: `alpha`: float
                    alpha coefficient for loss.
                key: `beta`: float
                    beta coefficient for loss.
            x_dimension: integer
                number of gene expression space dimensions.
            z_dimension: integer
                number of latent space dimensions.
    """

    def __init__(self, x_dimension, z_dimension=40, **kwargs):
        super().__init__()
        self.x_dimension = x_dimension
        self.z_dimension = z_dimension

        self.model_path = kwargs.get("model_path", "./models/")
        self.lambda_cycle = kwargs.get("lambda_cycle", 10.0)
        self.lambda_id = kwargs.get("lambda_id", 0.1)
        self.lambda_l1 = kwargs.get("lambda_l1", 0.0)
        self.lambda_l2 = kwargs.get("lambda_l2", 0.0)
        self.lr = kwargs.get("learning_rate", 0.001)
        self.dr_rate = kwargs.get("dropout_rate", 0.5)
        self.output_activation = kwargs.get("output_activation", 'relu')

        self.x = Input(shape=(self.x_dimension,), name="data")
        self.z = Input(shape=(self.z_dimension,), name='latent')

        self.init_w = keras.initializers.truncated_normal(stddev=0.02)
        self.regularizer = keras.regularizers.l1_l2(self.lambda_l1, self.lambda_l2)
        self.aux_models = {}

        self.__create_network()
        self.__compile_network()

    def __build_discriminator(self, name):
        h = Dense(700, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(self.x)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)

        h = Dense(100, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)

        h = Dense(1, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)

        discriminator = Model(self.x, h, name=name)

        return discriminator

    def __build_generator(self, name):
        h = Dense(700, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(self.x)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)

        h = Dense(100, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)

        h = Dense(self.z_dimension, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer,
                  use_bias=False)(h)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h_z = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h_z)

        h = Dense(100, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)

        h = Dense(700, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)

        h = Dense(self.x_dimension, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer,
                  use_bias=False)(h)
        h = BatchNormalization(axis=1, trainable=True)(h)
        h = ReLU()(h)

        generator = Model(self.x, h, name=name)
        self.aux_models[f"{name}_latent"] = Model(self.x, h_z, name=f"{name}_latent")
        return generator

    def __create_network(self):
        self.source_disc = self.__build_discriminator("disc_A")
        self.target_disc = self.__build_discriminator("disc_B")

        self.__compile_discriminators()

        self.g_AB = self.__build_generator("gen_AB")
        self.g_BA = self.__build_generator("gen_BA")

        x_A = Input(shape=(self.x_dimension,))
        x_B = Input(shape=(self.x_dimension,))

        fake_B = self.g_AB(x_A)
        fake_A = self.g_BA(x_B)

        x_hat_A = self.g_BA(fake_B)
        x_hat_B = self.g_AB(fake_A)

        img_A_id = self.g_BA(x_A)
        img_B_id = self.g_AB(x_B)

        self.source_disc.trainable = False
        self.target_disc.trainable = False

        valid_A = self.source_disc(fake_A)
        valid_B = self.source_disc(fake_B)

        optimizer = Adam(self.lr)
        self.combined = Model(inputs=[x_A, x_B],
                              outputs=[valid_A, valid_B,
                                       x_hat_A, x_hat_B,
                                       img_A_id, img_B_id])

        self.combined.compile(loss=['mse', 'mse',
                                    'mse', 'mse',
                                    'mse', 'mse'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

    def __compile_discriminators(self):
        optimizer = Adam(self.lr)
        self.source_disc.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])
        self.target_disc.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])

    def __compile_network(self):
        pass

    def to_latent(self, adata):
        adata = remove_sparsity(adata)

        latent = self.aux_models['gen_AB_latent'].predict(adata.X)
        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata

    def predict(self, adata, cell_type_to_predict, source_condition, condition_key, cell_type_key):
        adata = remove_sparsity(adata)

        cell_type_adata = adata[adata.obs[cell_type_key] == cell_type_to_predict]
        source_adata = cell_type_adata[cell_type_adata.obs[condition_key] == source_condition]

        reconstructed = self.g_AB.predict(source_adata.X)
        reconstructed_adata = anndata.AnnData(X=reconstructed)
        reconstructed_adata.obs = source_adata.obs.copy(deep=True)
        reconstructed_adata.var_names = cell_type_adata.var_names

        return reconstructed_adata



    def restore_model(self):
        self.source_disc = load_model(os.path.join(self.model_path, "source_disc.h5"), compile=False)
        self.target_disc = load_model(os.path.join(self.model_path, "target_disc.h5"), compile=False)
        self.g_AB = load_model(os.path.join(self.model_path, "g_AB.h5"), compile=False)
        self.g_BA = load_model(os.path.join(self.model_path, "g_BA.h5"), compile=False)
        self.combined = load_model(os.path.join(self.model_path, "combined.h5"), compile=False)

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.source_disc.save(os.path.join(self.model_path, "source_disc.h5"))
        self.target_disc.save(os.path.join(self.model_path, "target_disc.h5"))
        self.g_AB.save(os.path.join(self.model_path, "g_AB.h5"))
        self.g_BA.save(os.path.join(self.model_path, "g_BA.h5"))
        self.combined.save(os.path.join(self.model_path, "combined.h5"))

    def train(self, train_adata, valid_adata, condition_key, source_condition, target_condition, n_epochs=1000,
              batch_size=64):
        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))

        source_adata = train_adata[train_adata.obs[condition_key] == source_condition]
        target_adata = train_adata[train_adata.obs[condition_key] == target_condition]

        min_size = min(source_adata.shape[0], target_adata.shape[0])

        for epoch in range(n_epochs):
            d_loss = g_loss = d_acc = 0
            for batch_idx in range(min_size // batch_size):
                source_real_batch = source_adata.X[(batch_idx) * batch_size: (batch_idx + 1) * batch_size]
                target_real_batch = target_adata.X[(batch_idx) * batch_size: (batch_idx + 1) * batch_size]

                target_fake_batch = self.g_AB.predict(source_real_batch)
                source_fake_batch = self.g_BA.predict(target_real_batch)

                dA_loss_real = self.source_disc.train_on_batch(source_real_batch, valid)
                dA_loss_fake = self.source_disc.train_on_batch(source_fake_batch, fake)
                source_d_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.target_disc.train_on_batch(target_real_batch, valid)
                dB_loss_fake = self.target_disc.train_on_batch(target_fake_batch, fake)
                target_d_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                d_loss_batch = 0.5 * np.add(source_d_loss, target_d_loss)

                g_loss_batch = self.combined.train_on_batch([source_real_batch, target_real_batch],
                                                            [valid, valid,
                                                             source_real_batch, target_real_batch,
                                                             source_real_batch, target_real_batch])
                d_loss += d_loss_batch[0] / (min_size // batch_size)
                g_loss += np.mean(g_loss_batch) / (min_size // batch_size)

                d_acc += d_loss_batch[1] / (min_size // batch_size)

            print(f"Epoch {epoch}/{n_epochs}: ")
            print(f" - D_loss: {d_loss:.4f} - D_acc: {d_acc:.4f} - G_loss: {g_loss:.4f}")
