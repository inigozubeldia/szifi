## SZiFi, the Sunyaev-Zeldovich iterative Finder

SZiFi is a Python implementation of the iterative multi-frequency matched filter (iMMF) cluster-finding method, which is described in detail in [this paper]([https://arxiv.org/abs/2204.13780](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.4766Z/abstract)). It can be used to detect galaxy clusters with mm intensity maps through their thermal Sunyaev-Zeldovich (tSZ) signal. As a new feature, it allows for foreground deprojection via a spectrally constrained MMF, or sciMMF (see [this other paper](https://arxiv.org/abs/2212.07410](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5123Z/abstract)). It can also be used for point source detection. If you use SZiFi in any of your projects, please cite both papers.

SZiFi has been tested extensively on *Planck* simulated data, and its performance is currently being tested in the context of the Atacama Cosmology Telescope (ACT) and the Simons Observatory (SO).

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
Dependencies: [astropy](https://www.astropy.org), [healpy](https://healpy.readthedocs.io/en/latest/), [pymaster](https://namaster.readthedocs.io), [scikit-learn](https://scikit-learn.org/stable/).

### Sample scripts

Several sample scripts illustrating how the code works are included in szifi/test_files. In order to be able to run them, please download the data [here](https://drive.google.com/drive/folders/1_O48SQ5aPTaW32MAzBF6SEX7HyPvRoXM?usp=sharing) and put it in szifi/data.

