## SZiFi, the Sunyaev-Zeldovich iterative Finder

SZiFi (pronounced "sci-fi") is a Python implementation of the iterative multi-frequency matched filter (iMMF) galaxy cluster finding method, which is described in detail in [Zubeldia et al. (2023a)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.4766Z/abstract). It can be used to detect galaxy clusters with mm intensity maps through their thermal Sunyaev-Zeldovich (tSZ) signal. As a novel feature, it allows for foreground deprojection via a spectrally constrained MMF, or sciMMF (see [Zubeldia et al., 2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5123Z/abstract)). It can also be used for point source detection. If you use SZiFi in any of your projects, please cite both papers.

SZiFi has been used to produce a new set of [*Planck*](https://pla.esac.esa.int/#home) cluster catalogues (see [Zubeldia et al. (2024)](https://www.arxiv.org/abs/2408.06189)). Its peformance has been tested extensively on synthetic [*Planck*](https://pla.esac.esa.int/#home) data and is currently being tested in the context of the [Simons Observatory](https://simonsobservatory.org).

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

Several sample scripts illustrating how the code works are included in szifi/test_files. In order to be able to run them, please download the data [here](https://drive.google.com/drive/folders/1_O48SQ5aPTaW32MAzBF6SEX7HyPvRoXM?usp=sharing) and put it in a new directory called szifi/data.

