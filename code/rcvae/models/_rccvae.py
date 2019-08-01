import logging
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger, History, EarlyStopping
from keras.layers import Activation
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Conv2D, \
    Flatten, Reshape, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras_vggface.vggface import VGGFace
from scipy import sparse

from .utils import label_encoder
from ..data_loader import PairedDataSequence

log = logging.getLogger(__file__)


class RCCVAE:
    """
        Regularized Convolutional C-VAE vector Network class. This class contains the implementation of Conditional
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

    def __init__(self, x_dimension, z_dimension=100, **kwargs):
        # tf.reset_default_graph()
        self.x_dim = x_dimension if isinstance(x_dimension, tuple) else (x_dimension,)
        self.z_dim = z_dimension
        self.image_shape = x_dimension

        self.mmd_dim = kwargs.get("mmd_dimension", 128)
        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.beta = kwargs.get("beta", 100)
        self.gamma = kwargs.get("gamma", 1.0)
        self.conditions = kwargs.get("condition_list")
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "./")
        self.train_with_fake_labels = kwargs.get("train_with_fake_labels", False)
        self.kernel_method = kwargs.get("kernel", "multi-scale-rbf")
        self.arch_style = kwargs.get("arch_style", 1)
        self.n_gpus = kwargs.get("gpus", 1)

        self.x = Input(shape=self.x_dim, name="data")
        self.encoder_labels = Input(shape=(1,), name="encoder_labels")
        self.decoder_labels = Input(shape=(1,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        if self.x_dim[0] > 48 and self.gamma > 0:
            self.vggface = VGGFace(include_top=False, input_shape=self.x_dim, model='vgg16')
            self.vggface_layers = ["conv1_1", 'conv1_2',
                                   'conv2_1', 'conv2_2',
                                   'conv3_1', 'conv3_2', 'conv3_3']

        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function(compile_gpu_model=True)
        self.cvae_model.summary()

    def _encoder(self, name="encoder"):
        """
            Constructs the encoder sub-network of C-VAE. This function implements the
            encoder part of Variational Auto-encoder. It will transform primary
            data in the `n_vars` dimension-space to the `z_dimension` latent space.
            # Parameters
                No parameters are needed.
            # Returns
                mean: Tensor
                    A dense layer consists of means of gaussian distributions of latent space dimensions.
                log_var: Tensor
                    A dense layer consists of log transformed variances of gaussian distributions of latent space dimensions.
        """
        if self.arch_style == 1:  # Baseline CNN
            h = Dense(128, activation='relu')(self.encoder_labels)
            h = Dense(np.prod(self.x_dim[:-1]), activation='relu')(h)
            h = Reshape((*self.x_dim[:-1], 1))(h)
            h = concatenate([self.x, h])
            h = Conv2D(64, kernel_size=(4, 4), strides=2, padding='same')(h)
            h = LeakyReLU()(h)
            h = Conv2D(128, kernel_size=(4, 4), strides=2, padding='same')(h)
            h = LeakyReLU()(h)
            h = Flatten()(h)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
            log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
            z = Lambda(self._sample_z, output_shape=(self.z_dim,))([mean, log_var])
            model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
            model.summary()
            return mean, log_var, model
        elif self.arch_style == 2:  # FCN
            x_reshaped = Reshape(target_shape=(np.prod(self.x_dim),))(self.x)
            xy = concatenate([x_reshaped, self.encoder_labels], axis=1)
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(xy)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
            log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
            z = Lambda(self._sample_z, output_shape=(self.z_dim,))([mean, log_var])
            model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
            model.summary()
            return mean, log_var, model
        else:
            h = Dense(128, activation='relu')(self.encoder_labels)
            h = Dense(np.prod(self.x_dim[:-1]), activation='relu')(h)
            h = Reshape((*self.x_dim[:-1], 1))(h)
            h = concatenate([self.x, h])

            conv1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_1')(h)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_2')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1')(pool1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1')(pool2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2')(conv3)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_3')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_1')(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_2')(conv4)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_3')(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_1')(pool4)
            conv5 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_2')(conv5)
            conv5 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_3')(conv5)
            pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

            flat = Flatten(name='flatten')(pool5)

            dense = Dense(1024, activation='linear', name='fc1')(flat)
            dense = Activation('relu')(dense)

            dense = Dense(256, activation='linear', name='fc2')(dense)
            dense = Activation('relu')(dense)
            self.enc_dense = Dropout(self.dr_rate)(dense)

            mean = Dense(self.z_dim, kernel_initializer=self.init_w)(self.enc_dense)
            log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(self.enc_dense)

            z = Lambda(self._sample_z, output_shape=(self.z_dim,))([mean, log_var])
            model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
            # if self.x_dim[0] > 48:
            #     for layer_name in self.vggface_layers[1:]:
            #         model.get_layer(layer_name).set_weights(self.vggface.get_layer(layer_name).get_weights())
            model.summary()
            return mean, log_var, model

    def _mmd_decoder(self, name="decoder"):
        """
            Constructs the decoder sub-network of C-VAE. This function implements the
            decoder part of Variational Auto-encoder. It will transform constructed
            latent space to the previous space of data with n_dimensions = n_vars.
            # Parameters
                No parameters are needed.
            # Returns
                h: Tensor
                    A Tensor for last dense layer with the shape of [n_vars, ] to reconstruct data.
        """
        if self.arch_style == 1:  # Baseline CNN for MNIST
            zy = concatenate([self.z, self.decoder_labels], axis=1)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(zy)
            h = BatchNormalization(axis=1)(h)
            h_mmd = LeakyReLU(name="mmd")(h)
            h = Dense(np.prod(self.x_dim), kernel_initializer=self.init_w, use_bias=False)(h_mmd)
            h = LeakyReLU()(h)
            h = Reshape(target_shape=self.x_dim)(h)
            h = Conv2DTranspose(128, kernel_size=(4, 4), padding='same')(h)
            h = LeakyReLU()(h)
            h = Conv2DTranspose(64, kernel_size=(4, 4), padding='same')(h)
            h = LeakyReLU()(h)
            h = Conv2DTranspose(self.x_dim[-1], kernel_size=(4, 4), padding='same', activation="relu")(h)
            model = Model(inputs=[self.z, self.decoder_labels], outputs=[h, h_mmd], name=name)
            model.summary()
            return h, h_mmd, model
        elif self.arch_style == 2:  # FCN
            zy = concatenate([self.z, self.decoder_labels], axis=1)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(zy)
            h = BatchNormalization(axis=1)(h)
            h_mmd = LeakyReLU(name="mmd")(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(np.prod(self.x_dim), kernel_initializer=self.init_w, use_bias=True)(h)
            h = Activation('relu', name="reconstruction_output")(h)
            h = Reshape(target_shape=self.x_dim)(h)
            model = Model(inputs=[self.z, self.decoder_labels], outputs=[h, h_mmd], name=name)
            model.summary()
            return h, h_mmd, model
        else:
            encode_y = Dense(64, activation='relu')(self.decoder_labels)
            zy = concatenate([self.z, encode_y], axis=1)
            zy = Activation('relu')(zy)

            # zy = concatenate([zy, self.enc_dense], axis=1)
            # zy = Activation('relu')(zy)

            h = Dense(self.mmd_dim, activation="linear", kernel_initializer='he_normal')(zy)
            h_mmd = Activation('relu', name="mmd")(h)

            h = Dense(1024, kernel_initializer='he_normal')(h_mmd)
            h = Activation('relu')(h)

            h = Dense(256 * 4 * 4, kernel_initializer='he_normal')(h)
            h = Activation('relu')(h)

            width = self.x_dim[0] // 16
            height = self.x_dim[1] // 16
            n_channels = 4096 // (width * height)
            h = Reshape(target_shape=(width, height, n_channels))(h)

            up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(h))
            conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)

            up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv6))
            conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)

            up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv7))
            conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)

            up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv8))
            conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)

            conv10 = Conv2D(self.x_dim[-1], 1, activation='relu')(conv9)

            model = Model(inputs=[self.z, self.decoder_labels], outputs=[conv10, h_mmd],
                          name=name)
            model.summary()
            return h, h_mmd, model

    @staticmethod
    def _sample_z(args):
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed in `_encoder` function.
            # Parameters
                No parameters are needed.
            # Returns
                The computed Tensor of samples with shape [size, z_dim].
        """
        mu, log_var = args
        batch_size = K.shape(mu)[0]
        z_dim = K.int_shape(mu)[1]
        eps = K.random_normal(shape=[batch_size, z_dim])
        return mu + K.exp(log_var / 2) * eps

    def _create_network(self):
        """
            Constructs the whole C-VAE network. It is step-by-step constructing the C-VAE
            network. First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of C-VAE.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
        """

        inputs = [self.x, self.encoder_labels, self.decoder_labels]
        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        self.x_hat, self.mmd_hl, self.decoder_model = self._mmd_decoder(name="decoder")
        # if self.arch_style < 3:
        decoder_outputs = self.decoder_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])
        # else:
        #     decoder_outputs = self.decoder_model(
        #         [self.x, self.encoder_model(inputs[:2])[2], self.encoder_labels, self.decoder_labels])
        reconstruction_output = Lambda(lambda x: x, name="kl_reconstruction")(decoder_outputs[0])
        mmd_output = Lambda(lambda x: x, name="mmd")(decoder_outputs[1])
        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output],
                                name="cvae")
        if self.n_gpus > 1:
            self.gpu_cvae_model = multi_gpu_model(self.cvae_model,
                                                  gpus=self.n_gpus)
            self.gpu_encoder_model = multi_gpu_model(self.encoder_model,
                                                     gpus=self.n_gpus)
            self.gpu_decoder_model = multi_gpu_model(self.decoder_model,
                                                     gpus=self.n_gpus)
        else:
            self.gpu_cvae_model = self.cvae_model
            self.gpu_encoder_model = self.encoder_model
            self.gpu_decoder_model = self.decoder_model

    @staticmethod
    def compute_kernel(x, y, method='rbf', **kwargs):
        """
            Computes RBF kernel between x and y.
            # Parameters
                x: Tensor
                    Tensor with shape [batch_size, z_dim]
                y: Tensor
                    Tensor with shape [batch_size, z_dim]
            # Returns
                returns the computed RBF kernel between x and y
        """
        scales = kwargs.get("scales", [])
        if method == "rbf":
            x_size = K.shape(x)[0]
            y_size = K.shape(y)[0]
            dim = K.shape(x)[1]
            tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
            tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
            return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32))
        elif method == 'raphy':
            scales = K.variable(value=np.asarray(scales))
            squared_dist = K.expand_dims(RCCVAE.squared_distance(x, y), 0)
            scales = K.expand_dims(K.expand_dims(scales, -1), -1)
            weights = K.eval(K.shape(scales)[0])
            weights = K.variable(value=np.asarray(weights))
            weights = K.expand_dims(K.expand_dims(weights, -1), -1)
            return K.sum(weights * K.exp(-squared_dist / (K.pow(scales, 2))), 0)
        elif method == "multi-scale-rbf":
            sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

            beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
            distances = RCCVAE.squared_distance(x, y)
            s = K.dot(beta, K.reshape(distances, (1, -1)))

            return K.reshape(tf.reduce_sum(tf.exp(-s), 0), K.shape(distances)) / len(sigmas)

    @staticmethod
    def squared_distance(x, y):  # returns the pairwise euclidean distance
        r = K.expand_dims(x, axis=1)
        return K.sum(K.square(r - y), axis=-1)

    @staticmethod
    def compute_mmd(x, y, kernel_method, **kwargs):  # [batch_size, z_dim] [batch_size, z_dim]
        """
            Computes Maximum Mean Discrepancy(MMD) between x and y.
            # Parameters
                x: Tensor
                    Tensor with shape [batch_size, z_dim]
                y: Tensor
                    Tensor with shape [batch_size, z_dim]
            # Returns
                returns the computed MMD between x and y
        """
        x_kernel = RCCVAE.compute_kernel(x, x, method=kernel_method, **kwargs)
        y_kernel = RCCVAE.compute_kernel(y, y, method=kernel_method, **kwargs)
        xy_kernel = RCCVAE.compute_kernel(x, y, method=kernel_method, **kwargs)
        return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

    def _loss_function(self, compile_gpu_model=True):
        """
            Defines the loss function of C-VAE network after constructing the whole
            network. This will define the KL Divergence and Reconstruction loss for
            C-VAE and also defines the Optimization algorithm for network. The C-VAE Loss
            will be weighted sum of reconstruction loss and KL Divergence loss.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
        """

        def batch_loss():
            def perceptual_loss(input_image, reconstructed_image):
                vggface = VGGFace(include_top=False, input_shape=self.x_dim, model='vgg16')
                vgg_layers = ['conv1_1']
                outputs = [vggface.get_layer(l).output for l in vgg_layers]
                model = Model(inputs=vggface.input, outputs=outputs)

                for layer in model.layers:
                    layer.trainable = False

                input_image *= 255.0
                reconstructed_image *= 255.0

                input_image = preprocess_input(input_image, mode='tf', data_format='channels_last')
                reconstructed_image = preprocess_input(reconstructed_image, mode='tf', data_format='channels_last')

                h1_list = model(input_image)
                h2_list = model(reconstructed_image)

                if not isinstance(h1_list, list):
                    h1_list = [h1_list]
                    h2_list = [h2_list]

                p_loss = 0.0
                for h1, h2 in zip(h1_list, h2_list):
                    h1 = K.batch_flatten(h1)
                    h2 = K.batch_flatten(h2)
                    p_loss += K.mean(K.square(h1 - h2), axis=-1)

                return p_loss

            def kl_recon_loss(y_true, y_pred):
                y_pred = K.reshape(y_pred, (-1, *self.x_dim))
                y_true = K.reshape(y_true, (-1, *self.x_dim))

                kl_loss = 0.5 * K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, 1)
                recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=[1, 2, 3])
                if self.gamma > 0:
                    p_loss = perceptual_loss(y_true, y_pred)
                else:
                    p_loss = 0.0
                return self.alpha * kl_loss + self.gamma * p_loss + recon_loss

            def mmd_loss(real_labels, y_pred):
                y_pred = K.reshape(y_pred, (-1, self.mmd_dim))
                with tf.variable_scope("mmd_loss", reuse=tf.AUTO_REUSE):
                    real_labels = K.reshape(K.cast(real_labels, 'int32'), (-1,))
                    source_mmd, dest_mmd = tf.dynamic_partition(y_pred, real_labels, num_partitions=2)
                    loss = self.compute_mmd(source_mmd, dest_mmd, self.kernel_method)
                    return self.beta * loss

            self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr)
            if compile_gpu_model:
                self.gpu_cvae_model.compile(optimizer=self.cvae_optimizer,
                                            loss=[kl_recon_loss, mmd_loss],
                                            metrics={self.cvae_model.outputs[0].name: kl_recon_loss,
                                                     self.cvae_model.outputs[1].name: mmd_loss})
            else:
                self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                        loss=[kl_recon_loss, mmd_loss],
                                        metrics={self.cvae_model.outputs[0].name: kl_recon_loss,
                                                 self.cvae_model.outputs[1].name: mmd_loss})

        batch_loss()

    def to_latent(self, data, labels):
        """
            Map `data` in to the latent space. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
            # Returns
                latent: numpy nd-array
                    returns array containing latent space encoding of 'data'
        """
        latent = self.encoder_model.predict([data, labels])[2]
        return latent

    def to_mmd_layer(self, model, data, encoder_labels, feed_fake=False):
        """
            Map `data` in to the pn layer after latent layer. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
            # Returns
                latent: numpy nd-array
                    returns array containing latent space encoding of 'data'
        """
        if feed_fake:
            decoder_labels = np.ones(shape=encoder_labels.shape)
        else:
            decoder_labels = encoder_labels
        mmd_latent = model.cvae_model.predict([data, encoder_labels, decoder_labels])[1]
        return mmd_latent

    def _reconstruct(self, data, encoder_labels, decoder_labels):
        """
            Map back the latent space encoding via the decoder.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix whether in latent space or primary space.
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
                use_data: bool
                    this flag determines whether the `data` is already in latent space or not.
                    if `True`: The `data` is in latent space (`data.X` is in shape [n_obs, z_dim]).
                    if `False`: The `data` is not in latent space (`data.X` is in shape [n_obs, n_vars]).
            # Returns
                rec_data: 'numpy nd-array'
                    returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, n_vars].
        """
        latent = self.to_latent(data, encoder_labels)
        # if self.arch_style < 3:
        rec_data = self.decoder_model.predict([latent, decoder_labels])
        # else:
        #     rec_data = self.decoder_model.predict([data, latent, encoder_labels, decoder_labels])
        return rec_data

    def predict(self, data, encoder_labels, decoder_labels, data_space='None'):
        """
            Predicts the cell type provided by the user in stimulated condition.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix whether in primary space.
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
            # Returns
                stim_pred: numpy nd-array
                    `numpy nd-array` of predicted cells in primary space.
            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read("train_kang.h5ad")
            validation_data = sc.read("./data/validation.h5ad")
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            prediction = network.predict('CD4T', obs_key={"cell_type": ["CD8T", "NK"]})
            ```
        """
        if sparse.issparse(data.X):
            data.X = data.X.A

        input_data = np.reshape(data.X, (-1, *self.x_dim))

        if data_space == 'mmd':
            stim_pred = self._reconstruct_from_mmd(input_data)
        else:
            stim_pred = self._reconstruct(input_data, encoder_labels, decoder_labels)
        return stim_pred[0]

    def _reconstruct_from_mmd(self, data):
        model = Model(inputs=self.decoder_model.layers[1], outputs=self.decoder_model.outputs)
        return model.predict(data)

    def restore_model(self):
        """
            restores model weights from `model_to_use`.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read("./data/train_kang.h5ad")
            validation_data = sc.read("./data/valiation.h5ad")
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.restore_model()
            ```
        """
        self.cvae_model = load_model(os.path.join(self.model_to_use, 'mmd_cvae.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_to_use, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_to_use, 'decoder.h5'), compile=False)
        self._loss_function(compile_gpu_model=False)

    def train(self, train_data, use_validation=False, valid_data=None, n_epochs=25, batch_size=32,
              early_stop_limit=20,
              threshold=0.0025, initial_run=True,
              shuffle=True, verbose=2, save=True, paired=False):  # TODO: Write minibatches for each source and destination
        """
            Trains the network `n_epochs` times with given `train_data`
            and validates the model using validation_data if it was given
            in the constructor function. This function is using `early stopping`
            technique to prevent overfitting.
            # Parameters
                n_epochs: int
                    number of epochs to iterate and optimize network weights
                early_stop_limit: int
                    number of consecutive epochs in which network loss is not going lower.
                    After this limit, the network will stop training.
                threshold: float
                    Threshold for difference between consecutive validation loss values
                    if the difference is upper than this `threshold`, this epoch will not
                    considered as an epoch in early stopping.
                full_training: bool
                    if `True`: Network will be trained with all batches of data in each epoch.
                    if `False`: Network will be trained with a random batch of data in each epoch.
                initial_run: bool
                    if `True`: The network will initiate training and log some useful initial messages.
                    if `False`: Network will resume the training using `restore_model` function in order
                        to restore last model which has been trained with some training dataset.
            # Returns
                Nothing will be returned
            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read(train_katrain_kang.h5ad           >>> validation_data = sc.read(valid_kang.h5ad)
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            ```
        """
        if initial_run:
            log.info("----Training----")
        train_labels, _ = label_encoder(train_data)

        if sparse.issparse(train_data.X):
            train_data.X = train_data.X.A

        if use_validation and valid_data is None:
            raise Exception("valid_data is None but use_validation is True.")

        callbacks = [
            History(),
            EarlyStopping(patience=early_stop_limit, monitor='val_loss', min_delta=threshold),
            CSVLogger(filename="./csv_logger.log")
        ]
        if paired:
            xA_train = train_data[train_data.obs['condition'] == 0].X
            xB_train = train_data[train_data.obs['condition'] == 1].X

            xA_train = np.reshape(xA_train, newshape=(-1, *self.x_dim))
            xB_train = np.reshape(xB_train, newshape=(-1, *self.x_dim))

            x_train = np.concatenate([xA_train, xA_train, ], axis=0)
            y_train = np.concatenate([xA_train, xB_train, ], axis=0)
            encoder_labels_train = np.concatenate([np.zeros(xA_train.shape[0]), np.zeros(xA_train.shape[0]),
                                                   ])

            decoder_labels_train = np.concatenate([np.zeros(xA_train.shape[0]), np.ones(xA_train.shape[0]),
                                                   ])

            x = [x_train, encoder_labels_train, decoder_labels_train]
            y = [y_train, encoder_labels_train]

        else:
            x_train = np.reshape(train_data.X, newshape=(-1, *self.x_dim))
            x = [x_train, train_labels, train_labels]
            y = [x_train, train_labels]

        if use_validation:
            if paired:
                xA_test = valid_data[valid_data.obs['condition'] == 0].X
                xB_test = valid_data[valid_data.obs['condition'] == 1].X

                xA_test = np.reshape(xA_test, newshape=(-1, *self.x_dim))
                xB_test = np.reshape(xB_test, newshape=(-1, *self.x_dim))

                x_test = np.concatenate([xA_test, xA_test, ], axis=0)
                y_test = np.concatenate([xA_test, xB_test, ], axis=0)

                encoder_labels_test = np.concatenate([np.zeros(xA_test.shape[0]), np.zeros(xA_test.shape[0]),
                                                      ])

                decoder_labels_test = np.concatenate([np.zeros(xA_test.shape[0]), np.ones(xA_test.shape[0]),
                                                      ])

                x_test = [x_test, encoder_labels_test, decoder_labels_test]
                y_test = [y_test, encoder_labels_test]
            else:
                x_valid = np.reshape(valid_data.X, newshape=(-1, *self.x_dim))
                valid_labels, _ = label_encoder(valid_data)
                x_test = [x_valid, valid_labels, valid_labels]
                y_test = [x_valid, valid_labels]

            histories = self.gpu_cvae_model.fit(x=x,
                                                y=y,
                                                epochs=n_epochs,
                                                batch_size=batch_size,
                                                validation_data=(x_test, y_test),
                                                shuffle=shuffle,
                                                callbacks=callbacks,
                                                verbose=verbose,
                                                )
        else:
            histories = self.gpu_cvae_model.fit(x=x,
                                                y=y,
                                                epochs=n_epochs,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                callbacks=callbacks,
                                                verbose=verbose,
                                                )
        if save:
            os.makedirs(self.model_to_use, exist_ok=True)
            self.cvae_model.save(os.path.join(self.model_to_use, "mmd_cvae.h5"), overwrite=True)
            self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
            self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
            log.info(f"Model saved in file: {self.model_to_use}. Training finished")
        return histories

    def train_paired(self, train_path, use_validation=False, valid_path=None, n_epochs=25, batch_size=32,
                     early_stop_limit=20,
                     threshold=0.0025, initial_run=True,
                     shuffle=True, verbose=2, save=True):

        """
                    Trains the network `n_epochs` times with given `train_data`
                    and validates the model using validation_data if it was given
                    in the constructor function. This function is using `early stopping`
                    technique to prevent overfitting.
                    # Parameters
                        n_epochs: int
                            number of epochs to iterate and optimize network weights
                        early_stop_limit: int
                            number of consecutive epochs in which network loss is not going lower.
                            After this limit, the network will stop training.
                        threshold: float
                            Threshold for difference between consecutive validation loss values
                            if the difference is upper than this `threshold`, this epoch will not
                            considered as an epoch in early stopping.
                        full_training: bool
                            if `True`: Network will be trained with all batches of data in each epoch.
                            if `False`: Network will be trained with a random batch of data in each epoch.
                        initial_run: bool
                            if `True`: The network will initiate training and log some useful initial messages.
                            if `False`: Network will resume the training using `restore_model` function in order
                                to restore last model which has been trained with some training dataset.
                    # Returns
                        Nothing will be returned
                    # Example
                    ```python
                    import scanpy as sc
                    import scgen
                    train_data = sc.read(train_katrain_kang.h5ad           >>> validation_data = sc.read(valid_kang.h5ad)
                    network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
                    network.train(n_epochs=20)
                    ```
                """
        if initial_run:
            log.info("----Training----")

        callbacks = [
            History(),
            EarlyStopping(patience=early_stop_limit, monitor='val_loss', min_delta=threshold),
            CSVLogger(filename="./csv_logger.log")
        ]

        train_image_filepaths = [os.path.join(train_path, filepath) for filepath in os.listdir(train_path) if filepath.endswith('.jpg')]
        train_generator = PairedDataSequence(train_image_filepaths, batch_size=batch_size)
        if use_validation:
            valid_image_filepaths = [os.path.join(valid_path, filepath) for filepath in os.listdir(valid_path) if filepath.endswith('.jpg')]
            valid_generator = PairedDataSequence(valid_image_filepaths, batch_size=batch_size)
            histories = self.gpu_cvae_model.fit_generator(generator=train_generator,
                                                          steps_per_epoch=50,
                                                          workers=8,
                                                          use_multiprocessing=True,
                                                          epochs=n_epochs,
                                                          validation_data=valid_generator,
                                                          validation_steps=5,
                                                          shuffle=shuffle,
                                                          callbacks=callbacks,
                                                          verbose=verbose,
                                                          )
        else:
            histories = self.gpu_cvae_model.fit_generator(generator=train_generator,
                                                          steps_per_epoch=50,
                                                          workers=8,
                                                          use_multiprocessing=True,
                                                          epochs=n_epochs,
                                                          shuffle=shuffle,
                                                          callbacks=callbacks,
                                                          verbose=verbose,
                                                          )
        if save:
            os.makedirs(self.model_to_use, exist_ok=True)
            self.cvae_model.save(os.path.join(self.model_to_use, "mmd_cvae.h5"), overwrite=True)
            self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
            self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
            log.info(f"Model saved in file: {self.model_to_use}. Training finished")
        return histories