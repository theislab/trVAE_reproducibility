from ._cvae import CVAE
from ._trvae import trVAE
from ._mmdcvae import MMDCVAE
from ._cycle_gan import CycleGAN
from ._saucie import SAUCIE

MODELS = {
    "CycleGAN": CycleGAN,
    "CVAE": CVAE,
    "MMDCVAE": MMDCVAE,
    "trVAE": trVAE,
    "SAUCIE": SAUCIE
}

