import numpy as np

params_szifi_default = {

"theta_500_vec_arcmin": np.exp(np.linspace(np.log(0.5),np.log(15.),15)), #cluster search angular scales
"q_th": 4., #detection threshold
"q_th_noise": 4., #detection threshold to remove detections for iterative covariance estimation
"mask_radius": 3., #masking radius for iterative covariance estimation in units of theta_500 for each detection
"iterative": True, #if True, iterative noise covariance estimation
"max_it":1,

"estimate_spec": "estimate", #if "estimate", covariance is estimated from data; if "theory", it is computed theoretically
"decouple_type": "master", # "master", "none", or "fsky"
"save_coupling_matrix": False, #if True, the power spectra coupling matrix is saved (old save_mask)
"compute_coupling_matrix": False, #if True, the coupling matrix is computed from scratch, otherwise it is loaded

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
"true_cat_select":"q",

"theta_find": "input",
"detection_method": "maxima",
"apod_type": "old",
"path": "/Users/inigozubeldia/Desktop/szifi/",

"mmf_type": "standard", #"standard" or "spectrally_constrained"
"cmmf_type": "one_dep", #"one_dep" or "general"
"sed_b": None, #SED to be deprojected, if only one SED is deprojected. Otherwise, use "a_matrix"
"a_matrix": None, #SED matrix
"comp_to_calculate": [0], #Component in the mixing matrix to extract; if the tSZ SED is in the first column, this will be [0]

"save_snr_maps": False,
"snr_maps_name": None,
"snr_maps_path": None,

"get_lonlat": True,

"cosmology": "Planck15",
}

params_model_default = {
"profile_type":"arnaud", #"arnaud" or "point"
"concentration":1.177,
}


params_data_default = {
"data_set": "Planck_real",
"field_ids": [0],
"other_params":{
"components":["tSZ","kSZ","CIB","CMB","noise"],
}
}