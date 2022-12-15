## SZiFi, the Sunyaev-Zeldovich iterative Finder

SZiFi is a Python implementation of the iterative multi-frequency matched filter (iMMF) cluster-finding method, which is described in detail in https://arxiv.org/abs/2204.13780. It can be used to detect galaxy clusters with mm intensity maps through their thermal Sunyaev-Zeldovich (tSZ) signal. As a new feature, it allows for foreground deprojection via a spectrally constrained MMF, or sciMMF (see https://arxiv.org/abs/2212.07410). It can also be used for point-source detection.

SZiFi has been tested extensively on Planck-like simulated data, and its performance is currently being tested in the context of the upcoming Simons Observatory.

A set of tutorials on how to use it is currently in development.
