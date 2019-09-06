import anndata

from reptrvae.models import MODELS

MODELS_STRING = ["CycleGAN", "trVAE", "MMDCVAE", "trVAE", "scVI", "SAUCIE", "CVAE"]


def train(model: str,
          adata: anndata.AnnData,
          **kwargs):
    if not model in MODELS_STRING:
        raise Exception("Invalid model selected")

    model = MODELS[model](adata.shape[1])

    model.train(**kwargs)

    # TODO: complete this function



