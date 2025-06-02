## SZiFi, the Sunyaev-Zeldovich iterative Finder

SZiFi (pronounced "sci-fi") is a Python implementation of the iterative multi-frequency matched filter (iMMF) galaxy cluster finding method, described in detail in [Zubeldia et al. (2023a)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.4766Z/abstract) and [Zubeldia et al., 2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5123Z/abstract). It can be used to detect galaxy clusters in milimetre-wavelength intensity maps through their thermal Sunyaev-Zeldovich (tSZ) signal, as well as to detect point sources. If you use SZiFi in your work, please cite both papers.

Some SZiFi's key features include:

- It operates on sky cut-outs in which the flat-sky approximation is assumed to hold, natively supporting two sky tessellation schemes: one based on HEALPix pixels, suited for spherical maps in the HEALPix pixellation, and another one based on RA and Dec limits, suited for maps in the CAR projection.
- It allows for foreground deprojection (e.g., of the Cosmic Infrared Background) through its implementation of the spectrally constrained MMF, or sciMMF, method (see [Zubeldia et al., 2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5123Z/abstract)).
- It can incorporate the relativistic corrections to the tSZ SED through its interface with [SZpack](https://github.com/CMBSPEC/SZpack).
- It provides several noise covariance estimation methods: an isotropic method (for isotorpic noise) and multiple anisotropic methods (suitable for anisotropic noise); see `szifi/params.py` for details.

SZiFi has been used to produce a new set of [*Planck*](https://pla.esac.esa.int/#home) cluster catalogues (see [Zubeldia et al. 2024](https://ui.adsabs.harvard.edu/abs/2025MNRAS.tmp..470Z/abstract)). Its peformance has been tested extensively on synthetic *Planck* data and is currently being tested in the context of the [Simons Observatory](https://simonsobservatory.org) (SO).

If you have any questions about how to use the code, please write to me at inigo.zubeldia (at) ast cam ac uk.

### Installation

Download the source code and do 
```
$ pip install -e .
```
You'll then be able to import SZiFi in Python with
```
import szifi
```
Dependencies: [astropy](https://www.astropy.org), [healpy](https://healpy.readthedocs.io/en/latest/), [pymaster](https://namaster.readthedocs.io), [scikit-learn](https://scikit-learn.org/stable/) (optional), [SZpack](https://github.com/CMBSPEC/SZpack) (optional), [pixell](https://github.com/simonsobs/pixell/tree/master) (optional), [orphics](https://github.com/msyriac/orphics) (optional).

### Sample scripts

Several sample scripts illustrating how the code works on real Planck data and simulated SO-like data are included in `szifi/test_files`. In order to be able to run them, please download the data [here](https://drive.google.com/drive/folders/1_O48SQ5aPTaW32MAzBF6SEX7HyPvRoXM?usp=sharing) and put it in a new directory called `szifi/data`.

### 06/05/2025: Code upgrade

SZiFi has been significantly upgraded on 06/05/2025:

- It can now be applied to maps in the CAR projection through a new interface with the [pixell](https://github.com/simonsobs/pixell/tree/master) library, with tiles defined as RA and dec limits. See the example on test_files/run_szifi_so_car.py.
- The way the data is interfaced with the code has been simplified. Now, each dataset is defined through a survey data file. The two survey data files needed to run the test samples on `szifi/test_files` are provided in `szifi/surveys`.

### 30/08/2024: Code upgrade

SZiFi has been significantly upgraded on 30/08/2024. This upgrade includes the following:

- A significant performance boost and more efficient memory usage, thanks to Erik Rosenberg. Some of these improvements can be controlled with several new parameters (see params.py).
- Functionality to incorporate the tSZ relativistic corrections when performing the cluster extraction in the fixed mode. These corrections are computed using SZpack. We have included a new tutorial in test_files illustrating this new functionality.
- A fix of a minor bug when masking the detections outside the tessellation mask in iterative noise covariance estimation (thanks to Erik Rosenberg for noticing it).

We recommend that this latest version of the code is used. It should be compatible with code using the previous, original version.


