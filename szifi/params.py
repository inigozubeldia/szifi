import numpy as np
from copy import deepcopy

params_szifi_default = {

"theta_500_vec_arcmin": np.exp(np.linspace(np.log(0.5),np.log(15.),2)), #cluster search angular scales
"q_th": 4., #detection threshold
"q_th_noise": 4., #detection threshold to remove detections for iterative covariance estimation
"mask_radius": 3., #masking radius for iterative covariance estimation in units of theta_500 for each detection
"iterative": True, #if True, iterative noise covariance estimation
"max_it":1,

"decouple_type": "master", # "master", "none", or "fsky"
"save_coupling_matrix": False, #if True, the power spectra coupling matrix is saved (old save_mask)
"compute_coupling_matrix": False, #if True, the coupling matrix is computed from scratch, otherwise it is loaded
"coupling_matrix_needed": True, #whether the coupling matrix is needed (e.g., not needed in fsky or none decoupling)
"powspec_lmax1d": None, # maximum lmax for power spectra. Maps and masks will be degraded to this lmax before computing coupling matrix and power spectra
"powspec_new_shape": None, # New shape for calculating powspec on lower resolution map. One of this and powspec_lmax1d must be 'None'
"powspec_bin_fac": 4, # Factor by which to bin Cls. There will be nx/bin_fac bins of width bin_fac * (pi/(nx*dx)) for a field of nx^2 pixels
"cov_type": "isotropic", # "isotropic", "anisotropic_boxcar", or "anisotropic_gaussian"
"cov_kernel_shape": [4.,4.], #shape in pixels of the smoothing kernel (Gaussian or boxcar) for anisotropic covariance estimation
"snr_weigthing": False, #whether to weight the SNR maps by some weights (e.g., given by the hitcounts). If True, the weight map for each tile should be provided in the survey file

"lrange": [100,2500], #ell range to be used in the analysis, if None all the modes are used
"freqs": [0,1,2,3,4,5], #Frequency channels to be used in the analysis
"beam": "real", #"gaussian" or "real"

"interp_type": "nearest",
"n_inpaint": 100, #number of iterations for diffusive inpainting for point sources
"inpaint_type": "diffusive", #"diffusive", ""
"inpaint": True, #if True, point sources are inpainted
"lsep": 3000,
"max_radius_mask_arcmin": np.inf, #Maximum radius in arcmin to mask around clusters
"min_radius_mask_arcmin": 0., #Maximum radius in arcmin to mask around clusters. Should be informed by the beam widths.

"extraction_mode":"find", #"fixed" or "find"
"get_q_true":False, #has superseeded "extract_at_truth"
"theta_500_input": None,
"norm_type": "centre",
"min_ftile":0.3, #minimum unmasked fraction of tile for it to be considered for cluster finding
"tilemask_mode": "field", # "catalogue", "catalogue_buffer", or "field"
"tilemask_buffer_arcmin": 15,
"tile_type": "healpix", #"healpix" or "car"


"theta_find": "input",
"detection_method": "maxima_lomem",
"apod_type": "old",
"path": "/Users/inigozubeldia/Desktop/szifi/",
"path_data": "/Users/inigozubeldia/Desktop/szifi/data/",
"survey_file": "/Users/inigozubeldia/Desktop/szifi/surveys/data_planck.py",
"save_and_load_template": True,
"path_template": "/Users/inigozubeldia/Desktop/szifi/data/templates/",
"map_dtype": np.float32,

"mmf_type": "standard", #"standard" or "spectrally_constrained"
"cmmf_type": "one_dep", #"one_dep" or "general". If only one SED is deprojected, use "one_dep" (faster); they are mathematically the same
"a_matrix": None, #n_freq x n_component SED matrix. The first column should be the tSZ SED, the second column the deprojection SED
"comp_to_calculate": [0], #Component in the mixing matrix to extract; if the tSZ SED is in the first column, this will be [0]
"deproject_cib": None, #None, or ["cib"], ["cib","beta"], ["cib","betaT"], or ["cib","beta","betaT"; it superseeds the "a_matrix" given by hand
"integrate_bandpass":True,

"save_snr_maps": False,
"snr_maps_name": None,
"snr_maps_path": None,

"get_lonlat": True,

"cosmology": "Planck15",
"cosmology_tool": "classy_sz", #Only relevant if "cosmology" is "cosmocnc"

"rSZ": False,
}


params_model_default = {
#Pressure profile parameters
"profile_type":"arnaud", #"arnaud" or "point"
"concentration":1.177,

#CIB parameters
"alpha_cib":0.2, #Websky value
"T0_cib":20.7, #Websky value
"beta_cib":1.6, #Websky value
"gamma_cib":1.8, #Websky value
"z_eff_cib":0.2, #Used for Planck analysis
}


params_data_default = {
"data_set": "Planck_real",
"field_ids": [0],
"other_params":{
"components":["tSZ","kSZ","CIB","CMB","noise"],
}
}
