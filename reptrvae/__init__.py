"""ReptrVAE - Reproducibility package for Regularized Conditional Variational Autoencoders"""

from . import utils as tl
from . import models
from . import plotting as pl
from . import data_loader as dl

__author__ = ', '.join([
    'Mohsen Naghipourfar',
    'Mohammad Lotfollahi'
])

__email__ = ', '.join([
    'mohsen.naghipourfar@gmail.com',
    'Mohammad.lotfollahi@helmholtz-muenchen.de',
])

from get_version import get_version
__version__ = get_version(__file__)

del get_version





