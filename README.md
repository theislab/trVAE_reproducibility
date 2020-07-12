# trVAE_reproducibility

<img align="center" src="./sketch.png?raw=true">

## Getting Started

```bash
cd scripts/
python DataDownloader.py
python ModelTrainer.py all
```

Then you can run each notebook and reproduce the results.

All datasets are available in this drive [directory](https://drive.google.com/drive/folders/1n1SLbXha4OH7j7zZ0zZAxrj_-2kczgl8?usp=sharing).


# Running scripts

You can simply train each network with a specific dataset with the following scripts: 

## Train trVAE with Kang or Haber dataset
```bash
python -m scripts.train_trVAE kang[haber] 
```

## Train DCtrVAE with Morpho-MNIST or CelebA dataset
```bash
python -m scripts.train_DCtrVAE mnist[celeba] 
```

## Train CVAE with Kang or Haber dataset
```bash
python -m scripts.train_cvae kang[haber] 
```

## Train CycleGAN with Kang or Haber dataset
```bash
python -m scripts.train_cyclegan kang[haber]
```

## Train MMD-CVAE with Kang or Haber dataset
```bash
python -m scripts.train_mdcvae kang[haber]
```

## Train SAUCIE with Kang or Haber dataset
```bash
python -m scripts.train_saucie kang[haber]
```

## Train scGen with Kang or Haber dataset
```bash
python -m scripts.train_scGen kang[haber]
```

## Train scVI with Kang or Haber dataset
```bash
python -m scripts.train_scVI kang[haber]
```

# Table of Notebooks 


## Data Analysis
Study       | notebook path     
---------------| ---------------
| [*Haber et. al*](https://nbviewer.jupyter.org/github/Naghipourfar/trVAE_reproducibility/blob/master/Jupyter%20Notebooks/Haber.ipynb)| Jupyter Notebooks/Haber.ipynb| 
| [*Kang et. al*](https://nbviewer.jupyter.org/github/Naghipourfar/trVAE_reproducibility/blob/master/Jupyter%20Notebooks/Kang.ipynb)| Jupyter Notebooks/Kang.ipynb| 
| [*CelebA*](https://nbviewer.jupyter.org/github/Naghipourfar/trVAE_reproducibility/blob/master/Jupyter%20Notebooks/CelebA.ipynb)| Jupyter Notebooks/CelebA.ipynb| 

## Paper Plots 
Figures  | notebook path     
---------------| ---------------
| [*Method Comparison - Haber et. al*](https://nbviewer.jupyter.org/github/theislab/trVAE_reproducibility/blob/master/Jupyter%20Notebooks/methodComparison-Haber.ipynb)| Jupyter Notebooks/methodComparison-Haber.ipynb| 
| [*Method Comparison - Kang et. al*](https://nbviewer.jupyter.org/github/theislab/trVAE_reproducibility/blob/master/Jupyter%20Notebooks/methodComparison-Kang.ipynb)| Jupyter Notebooks/methodComparison-Kang.ipynb| 
| [*Runtime Comparison - Kang et. al*](https://nbviewer.jupyter.org/github/theislab/trVAE_reproducibility/blob/master/Jupyter%20Notebooks/Time.ipynb)| Jupyter Notebooks/Time.ipynb|
| [*Simulation Response - Kang et. al*](https://nbviewer.jupyter.org/github/theislab/trVAE_reproducibility/blob/master/Jupyter%20Notebooks/BoxPlots_StackedViolins%20-%20Kang.ipynb)| Jupyter Notebooks/BoxPlots_StackedViolins - Kang.ipynb|


To run the notebooks and scripts you need following packages :

tensorflow, scanpy, numpy, matplotlib, scipy, wget.
