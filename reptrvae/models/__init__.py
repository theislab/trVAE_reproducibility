from ._cvae import CVAE
from ._trvae import trVAE
from ._mmdcvae import MMDCVAE
from ._cycle_gan import CycleGAN
from ._saucie import SAUCIE_BACKEND
from ._scgen import scGen
from ._dctrvae import DCtrVAE

MODELS = {
    "CycleGAN": CycleGAN,
    "CVAE": CVAE,
    "MMDCVAE": MMDCVAE,
    "trVAE": trVAE,
    "SAUCIE": SAUCIE_BACKEND,
    "scGen": scGen,
}

