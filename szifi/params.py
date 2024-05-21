import numpy as np
from copy import deepcopy

params_szifi_default = {

"theta_500_vec_arcmin": np.exp(np.linspace(np.log(0.5),np.log(15.),2)), #cluster search angular scales
"q_th": 4., #detection threshold
"q_th_noise": 4., #detection threshold to remove detections for iterative covariance estimation
"mask_radius": 3., #masking radius for iterative covariance estimation in units of theta_500 for each detection
"iterative": True, #if True, iterative noise covariance estimation
"max_it":1,

"estimate_spec": "estimate", #if "estimate", covariance is estimated from data; if "theory", it is computed theoretically
"decouple_type": "master", # "master", "none", or "fsky"
"save_coupling_matrix": False, #if True, the power spectra coupling matrix is saved (old save_mask)
"compute_coupling_matrix": False, #if True, the coupling matrix is computed from scratch, otherwise it is loaded
"powspec_lmax1d": None, # maximum lmax for power spectra. Maps and masks will be degraded to this lmax before computing coupling matrix and power spectra
"powspec_new_shape": None, # New shape for calculating powspec on lower resolution map. One of this and powspec_lmax1d must be 'None'
"powspec_bin_fac": 4, # Factor by which to bin Cls. There will be nx/bin_fac bins of width bin_fac * (pi/(nx*dx)) for a field of nx^2 pixels

"lrange": [100,2500], #ell range to be used in the analysis, if None all the modes are used
"freqs": [0,1,2,3,4,5], #Frequency channels to be used in the analysis
"beam": "real", #"gaussian" or "real"

"interp_type": "nearest",
"n_inpaint": 100, #number of iterations for diffusive inpainting for point sources
"inpaint": True, #if True, point sources are inpainted
"lsep": 3000,

"extraction_mode":"find", #"fixed" or "find"
"get_q_true":False, #has superseeded "extract_at_truth"
"theta_500_input": None,
"norm_type": "centre",
"n_clusters_true": 1000, #maximum number of true clusters for which to extract signal
"min_ftile":0.3, #minimum unmasked fraction of tile for it to be considered for cluster finding

"theta_find": "input",
"detection_method": "maxima",
"apod_type": "old",
"path": "/global/homes/r/rosenber/Programs/szifi/",
"path_data": "/global/homes/r/rosenber/Programs/szifi/data/",

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
}

params_szifi_so = deepcopy(params_szifi_default)
params_so = {
    "beam" : "gaussian",
    "min_ftile" : 0.2,
    "lrange" : [100, 10000],
    "powspec_lmax1d": 10000}
for key in params_so.keys(): params_szifi_so[key] = params_so[key]

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
params_data_so = {
"data_set": "so_sims",
"field_ids": [0],
"other_params":{
"components":["tSZ","kSZ","CIB","CMB","noise"],
}
}
