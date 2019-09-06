import logging
import os

import anndata
import keras
import numpy as np
from keras.callbacks import CSVLogger, History, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import to_categorical
from scipy import sparse

from reptrvae.models._network import Network

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

    def __create_network(self):
        pass

    def __compile_network(self):
        pass

    def to_latent(self):
        pass

    def predict(self):
        pass

    def train(self):
        pass

