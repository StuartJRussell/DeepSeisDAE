# DeepSeisDAE

## Overview

This repository exists to document and preserve the deep-learning denoising auto-encoder (DAE) as implemented by Russell et al. (2026), which used a DAE  to improve the signal-to-noise ratio of teleseismic data, specifically Sdiff, for deep Earth study. This repository is based on that used by Heuel & Friedrich (2022) to remove wind turbine noise, with some modifications, additions, and adaptations for teleseismic applications. For more details on the methodology, please see Russell et al. (2026) and the other references listed at the bottom of this README.

In this repository we provide the code, documentation and example scripts that would allow reproduction of the models of Russell et al. (2026), as well as generation of similar models for those who wish to denoise their own teleseismic data. There may be further developments to this repository in the future. We note that a DAE is now incorporated in SeisBench.

## Getting started

### Setup

The DAE was built using Python3 through the Tensorflow software library. The file `environment.yml` defines a conda environment that contans everything needed to run the codes in this repository. To create the environment: `conda env create -f environment.yml`. See the conda documentation for further information.

### Training

All of the functions and classes required to train and use the DAE, can be found in the `/DAE` directory, which can be imported as a package. Documentation for specific functions and classes can be found in the files in this directory. `run_training.py` and `run_retraining.py` are scripts that run the training and retraining, respectively. Parameters for these are specified in `parfile` and `retrain_parfile`, respectively. Training can be run on both CPUs and GPUs.

Russell et al. (2026) trained the DAE using Instaseis synthetics as training signal, and noise from real seismic stations as training noise. The former was downloaded by accessing pre-computed Green's function databases hosted by Syngine using the Instaseis Python package; this is facilitated by the `DAE.SyntheticsDownloader` class and example usage can be found in the Jupyter-Notebook `example_synthetics.ipynb`. Noise was downloaded using the ObsPy FDSN webservice; this is facilitated by the `DAE.NoiseDownloader` class and example usage can be similarly found in the Jupyter-Notebook `example_noise.ipynb`.

### Denoising

The function `DAE.denoise` uses trained models to denoise seismic data. Seismic traces passed to this function must have the same number of samples and sampling rate as the traces used to train the model. Documentation for this function can be found in `DAE.denoinsing.py`.

## License

This repository is derived from the seisDAE repository developed by Janis Heuel and released under the GNU General Public License v3 (GPLv3). Accordingly, this repository is also distributed under the GNU General Public License v3 (GPLv3). A copy of the GPLv3 licence is provided in LICENSE. 

The original repository was developed for suppression of wind-turbine noise from seismological data and this repository contains modifications, additions, and removals to adapt the existing code to denoising teleseismic data for deep-Earth studies. A link to the seisDAE repository is given below.

## References and links

For this repository:
 * Russell, S., Hannemann, K., Martin, C., Lindhorst, B., Esmail, W. and Thomas, C., 2026. Denoising seismic data for deep Earth studies using a supervised deep-learning auto-encoder: a case study of diffracted waves from ULVZs. Journal of Geophysical Research: Solid Earth [accepted]

Denoising auto-encoder:
 * Zhu, W., Mousavi, S.M. and Beroza, G.C., 2019. Seismic signal denoising and decomposition using deep neural networks. IEEE Transactions on Geoscience and Remote Sensing, 57(11), pp.9476-9488.
 * Tibi, R., Hammond, P., Brogan, R., Young, C.J. and Koper, K., 2021. Deep learning denoising applied to regional distance seismic data in Utah. Bulletin of the Seismological Society of America, 111(2), pp.775-790.
 * Heuel, J. and Friederich, W., 2022. Suppression of wind turbine noise from seismological data using nonlinear thresholding and denoising autoencoder. Journal of Seismology, 26(5), pp.913-934.

Instaseis:
 * van Driel, M., Krischer, L., Stähler, S.C., Hosseini, K. and Nissen-Meyer, T., 2015. Instaseis: instant global seismograms based on a broadband waveform database. Solid Earth, 6(2), pp.701-717.

Seisbench:
 * Woollam, J., Münchmeyer, J., Tilmann, F., Rietbrock, A., Lange, D., Bornstein, T., Diehl, T., Giunchi, C., Haslinger, F., Jozinović, D. and Michelini, A., 2022. SeisBenchA toolbox for machine learning in seismology. Seismological Society of America, 93(3), pp.1695-1709.

ObsPy:
 * Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y. and Wassermann, J., 2010. ObsPy: A Python toolbox for seismology. Seismological Research Letters, 81(3), pp.530-533.

Links:
 * https://github.com/JanisHe/seisDAE
 * https://instaseis.net/
 * https://ds.iris.edu/ds/products/syngine/
 * https://docs.obspy.org/
 * https://docs.obspy.org/packages/obspy.clients.fdsn.html
 * https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
 * https://github.com/seisbench/seisbench
