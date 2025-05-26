import numpy as np
import szifi
import time

#Sample programme that runs SZiFi on synthetic Simons-Observatory-like sky cutout in the
#CAR projection as implemented in the pixell library (see szifi/surveys/data_so_car.py,
#where the data is defined.) SZiFi is first run in the cluster-finding mode, finding
#clusters blindly, and the cluster signals are then re-extracted at the recovered sky 
#positions and angular scales in order to illustrate SZiFi's fixed mode.

params_szifi = {
#"theta_500_vec_arcmin": np.exp(np.linspace(np.log(0.1),np.log(15.),20)), #cluster search angular scales
# "theta_500_vec_arcmin": np.exp(np.linspace(np.log(0.1),np.log(15.),20))[np.array([0, 4, 9, 14, 19])], #cluster search angular scales
"theta_500_vec_arcmin": np.array([2.,3.]), #cluster search angular scales
"q_th": 4.5, #detection threshold
"q_th_noise": 4., #detection threshold to remove detections for iterative covariance estimation
"mask_radius": 3., #masking radius for iterative covariance estimation in units of theta_500 for each detection
"iterative": True, #if True, iterative noise covariance estimation
"max_it":1,
"estimate_spec": "estimate", #if "estimate", covariance is estimated from data; if "theory", it is computed theoretically
"decouple_type": "master", # "master", "none", or "fsky"
"save_coupling_matrix": True, #if True, the power spectra coupling matrix is saved (old save_mask)
"compute_coupling_matrix": False, #if True, the coupling matrix is computed from scratch, otherwise it is loaded
"coupling_matrix_needed": True,
"powspec_lmax1d": 10000, # maximum lmax for power spectra. Maps and masks will be degraded to this lmax before computing coupling matrix and power spectra
"powspec_new_shape": None, # New shape for calculating powspec on lower resolution map. One of this and powspec_lmax1d must be 'None'
"powspec_bin_fac": 4, # Factor by which to bin Cls. There will be nx/bin_fac bins of width bin_fac * (pi/(nx*dx)) for a field of nx^2 pixels
"cov_type": "anisotropic_gaussian",#"anisotropic_boxcar",# "isotropic", "anisotropic_boxcar", or "anisotropic_gaussian"
"cov_kernel_shape": [3,3],
"lrange": [1,300000], #ell range to be used in the analysis, if None all the modes are used
"freqs": [0,1,2,3], #Frequency channels to be used in the analysis
"beam": "gaussian", #"gaussian" or "real"
"interp_type": "nearest",
"n_inpaint": 100, #number of iterations for diffusive inpainting for point sources
"inpaint": True, #if True, point sources are inpainted
"lsep": 12000,
"extraction_mode":"find", #"fixed" or "find"
"get_q_true":False, #has superseeded "extract_at_truth"
"theta_500_input": None,
"norm_type": "centre",
"min_ftile":0.2, #minimum unmasked fraction of tile for it to be considered for cluster finding
"tilemask_mode": "catalogue",
"tilemask_buffer_arcmin": 30,
"tile_type": "car", #"healpix" or "car"
"inpaint_type": "orphics",
"max_radius_mask_arcmin": np.inf,

"theta_find": "input",
"detection_method": "maxima_lomem",
"apod_type": "old",
"path": "/home/iz221/szifi/",
"path_data": "/home/iz221/szifi/data/", #Change this to your path to the data.
"survey_file": "/home/iz221/szifi/surveys/data_so_car.py", #Change this to your path to the survey file.
"save_and_load_template": False,
"path_template": None,
"map_dtype": np.float32,

"mmf_type": "standard", #"standard" or "spectrally_constrained"
"cmmf_type": None, #"one_dep" or "general". If only one SED is deprojected, use "one_dep" (faster); they are mathematically the same
"a_matrix": None, #n_freq x n_component SED matrix. The first column should be the tSZ SED, the second column the deprojection SED
"comp_to_calculate": [0], #Component in the mixing matrix to extract; if the tSZ SED is in the first column, this will be [0]
"deproject_cib": None, #None, or ["cib"], ["cib","beta"], ["cib","betaT"], or ["cib","beta","betaT"; it superseeds the "a_matrix" given by hand
"integrate_bandpass":False,
"save_snr_maps": False,
"snr_maps_name": "",
"snr_maps_path": "",
"get_lonlat": True,
"cosmology": "Websky",
"cosmology_tool": "classy_sz",

"rSZ": False,
}

field_ids = ['1_0_1']

params_data = {
"data_set": "so_sims",
"field_ids": field_ids,
}

params_model = {
#Pressure profile parameters
"profile_type":"arnaud", #"arnaud" or "point"
"concentration":1.177,
#CIB parameters, only relevant if spectral deprojection is applied
"alpha_cib":0.36,
"T0_cib":24.4, #same
"beta_cib":1.75, #same
"gamma_cib":1.7, #same
"z_eff_cib":0.2,
}

#Input data

t0 = time.time()

data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

#Find clusters

cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data,rank=0)
cluster_finder.find_clusters()

#Retrieve results

results = cluster_finder.results_dict
print("Time",time.time()-t0)

#Re-extract (to test the fixed mode)

detection_processor = szifi.detection_processor(results,params_szifi)
catalogue_obs_noit = detection_processor.results.catalogues["catalogue_find_0"]
catalogue_obs_it = detection_processor.results.catalogues["catalogue_find_1"]

#Re-extract in the fixed mode using as input catalogue the output catalogue of the blind search

print("Testing fixed mode")

data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

data.data["catalogue_input"] = {}
data.data["catalogue_input"][field_ids[0]] = catalogue_obs_it
params_szifi["extraction_mode"] = "fixed"

cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data)
cluster_finder.find_clusters()

results = cluster_finder.results_dict

detection_processor = szifi.detection_processor(results,params_szifi)

catalogue_obs_it_fixed = detection_processor.results.catalogues["catalogue_fixed_1"]

print("Blind SNRs",catalogue_obs_it.catalogue["q_opt"])
print("Re-extracted SNRs",catalogue_obs_it_fixed.catalogue["q_opt"])

print("As expected, the re-extracted SNRs are the same as those in the output catalogue of the blind search.")
