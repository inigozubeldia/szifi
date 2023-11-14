## SZiFi, the Sunyaev-Zeldovich iterative Finder

SZiFi is a Python implementation of the iterative multi-frequency matched filter (iMMF) cluster-finding method, which is described in detail in [this paper](https://arxiv.org/abs/2204.13780). It can be used to detect galaxy clusters with mm intensity maps through their thermal Sunyaev-Zeldovich (tSZ) signal. As a new feature, it allows for foreground deprojection via a spectrally constrained MMF, or sciMMF (see this [other paper](https://arxiv.org/abs/2212.07410) ). It can also be used for point source detection.

SZiFi has been tested extensively on *Planck* simulated data, and its performance is currently being tested in the context of the Atacama Cosmology Telescope (ACT) and the Simons Observatory (SO).

### Installation

Download the source code and do 
```
$ pip install -e .
```
You'll then be able to import SZiFi in Python with
```
import szifi
```
### Sample scripts

Several sample scripts illustrating how the code works are included in szifi/test_files. In order to be able to run them, please download the data [here](https://drive.google.com/drive/folders/1_O48SQ5aPTaW32MAzBF6SEX7HyPvRoXM?usp=sharing) and put it in szifi/data.

