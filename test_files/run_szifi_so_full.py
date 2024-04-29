import numpy as np
from mpi4py import MPI
import szifi
import time
#import tracemalloc
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print(rank)
#tracemalloc.start()
rank=0

def calculate_group_info(igroup, ngroup, imin, imax):
    Ntile = imax - imin
    w0, rr = Ntile//ngroup, Ntile % ngroup
    gwidth = [w0]*(ngroup-rr) + [w0+1]*rr
    start = np.concatenate(([0], np.cumsum(gwidth)[:-1])) + imin
    imin0 = start[igroup]
    imax0 = imin0 + gwidth[igroup]
    return imin0, imax0

MAXRANK = 1
NFIELDS = 1#536
imin = 208 # lowest field

if rank < MAXRANK:
    #Select fields
    n_core = MAXRANK
    all_field_ids = np.arange(*calculate_group_info(rank, MAXRANK, imin, imin+NFIELDS))
    print("Field ids",all_field_ids)
    #Iterate over different cases
    suffixes = [
    "mmf_6",
    ]
    mmf_types = [
    "standard",
    ]
    cmmf_types = [
    None,
    ]
    frequencies = [
    [0,1,2,3,4,5],
    ]
    dep_types = [
    None,
    ]
    for field_id in all_field_ids:
        field_ids = [field_id] # Put this in a loop to not load all data products at once
        for i in range(0,len(suffixes)):
            t0 = time.time()
            print(i)
            #Set parameters
            params_szifi = {
            #"theta_500_vec_arcmin": np.exp(np.linspace(np.log(0.1),np.log(15.),20)), #cluster search angular scales
            "theta_500_vec_arcmin": np.exp(np.linspace(np.log(0.1),np.log(15.),1)), #cluster search angular scales
            "q_th": 4., #detection threshold
            "q_th_noise": 4., #detection threshold to remove detections for iterative covariance estimation
            "mask_radius": 3., #masking radius for iterative covariance estimation in units of theta_500 for each detection
            "iterative": True, #if True, iterative noise covariance estimation
            "max_it":1,
            "estimate_spec": "estimate", #if "estimate", covariance is estimated from data; if "theory", it is computed theoretically
            "decouple_type": "master", # "master", "none", or "fsky"
            "save_coupling_matrix": False, #if True, the power spectra coupling matrix is saved (old save_mask)
            "compute_coupling_matrix": False, #if True, the coupling matrix is computed from scratch, otherwise it is loaded
            "powspec_lmax1d": 10000, # maximum lmax for power spectra. Maps and masks will be degraded to this lmax before computing coupling matrix and power spectra
            "powspec_new_shape": None, # New shape for calculating powspec on lower resolution map. One of this and powspec_lmax1d must be 'None'
            "powspec_bin_fac": 4, # Factor by which to bin Cls. There will be nx/bin_fac bins of width bin_fac * (pi/(nx*dx)) for a field of nx^2 pixels
            "lrange": [100,10000], #ell range to be used in the analysis, if None all the modes are used
            "freqs": frequencies[i], #Frequency channels to be used in the analysis
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
            "theta_find": "input",
            "detection_method": "maxima",
            "apod_type": "old",
            "path": "/nvme/scratch/erosen/programs/szifi/",
            "path_data": "/nvme/scratch/erosen/data/so_sims/",
            "mmf_type": mmf_types[i], #"standard" or "spectrally_constrained"
            "cmmf_type": cmmf_types[i], #"one_dep" or "general". If only one SED is deprojected, use "one_dep" (faster); they are mathematically the same
            "a_matrix": None, #n_freq x n_component SED matrix. The first column should be the tSZ SED, the second column the deprojection SED
            "comp_to_calculate": [0], #Component in the mixing matrix to extract; if the tSZ SED is in the first column, this will be [0]
            "deproject_cib": dep_types[i], #None, or ["cib"], ["cib","beta"], ["cib","betaT"], or ["cib","beta","betaT"; it superseeds the "a_matrix" given by hand
            "integrate_bandpass":False,
            "save_snr_maps": False,
            "snr_maps_name": None,
            "snr_maps_path": None,
            "get_lonlat": True,
            "cosmology": "Websky",
            }
            params_data = {
            "data_set": "so_sims",
            "field_ids": field_ids,
            }
            params_model = {
            #Pressure profile parameters
            "profile_type":"arnaud", #"arnaud" or "point"
            "concentration":1.177,
            #CIB parameters
            "alpha_cib":0.36, #from https://arxiv.org/pdf/1309.0382.pdf
            "T0_cib":24.4, #same
            "beta_cib":1.75, #same
            "gamma_cib":1.7, #same
            "z_eff_cib":0.2,
            }
            #Input data
            data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)
            #Find clusters
            cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data,rank=rank)
            cluster_finder.find_clusters()
            #Retrieve results
            results = cluster_finder.results_dict
            name = "planck_it_find_apodold_planckcibparams_" + suffixes[i]
            np.save("/nvme/scratch/erosen/data/so_sims/catalogues_szifi_so/" + name + "_" + str(rank) + ".npy",results)
            print("Time",time.time()-t0)
