import szifi
import numpy as np
import pylab as pl

#Sample programme that runs SZiFi on Planck data for two sky tiles with CIB deprojection.
#It assumes that the mask coupling matrix
#has already been calculated (see get_coupling_matrix.py)

#Set parameters

params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

params_szifi["mmf_type"] = "spectrally_constrained"
params_szifi["cmmf_type"] = "general"
params_szifi["cmmf_type"] = "general"
params_szifi["integrate_bandpass"] = True

#Input data

params_data["field_ids"] = [0,1]
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

#Deprojection data

freqs = params_szifi["freqs"]

params_model["alpha_cib"] = 0.36 #also from Planck paper
params_model["T0_cib"] = 20.7 # 24.4, this from Planck paper (https://arxiv.org/pdf/1309.0382.pdf), instead of 1.6
params_model["beta_cib"] = 1.6 # 1.75, this same
params_model["z_eff_cib"] = 0.2

#Compute CIB SED and its moments

cib_model = szifi.cib_model(params_model=params_model)
cib_sed = cib_model.get_sed_muK_experiment(experiment=data.data["experiment"],
bandpass=params_szifi["integrate_bandpass"])
cib_model.get_sed_first_moments_experiment(experiment=data.data["experiment"],
bandpass=params_szifi["integrate_bandpass"],moment_parameters=["beta","betaT"])

#Deprojecting CIB SED

a_matrix = np.zeros((len(freqs),2))
a_matrix[:,0] = data.data["experiment"].tsz_sed[freqs]
a_matrix[:,1] = cib_sed[freqs]
params_szifi["a_matrix"] = a_matrix

#Alternatively, deprojecting one moment (can choose between "betaT" and "beta"), comment out if only the SED is to be deprojected

a_matrix = np.zeros((len(freqs),3))
a_matrix[:,0] = data.data["experiment"].tsz_sed[freqs]
a_matrix[:,1] = cib_sed[freqs]
a_matrix[:,2] = cib_model.moments["betaT"]

#Set mixing matrix

params_szifi["a_matrix"] = a_matrix

#Alternatively, the CIB and its moments can be deprojected without the need to explicitly set the mixing matrix "a_matrix":

#params_szifi["deproject_cib"] = ["cib","betaT"] #Deprojecting the CIB SED and its first-order moment with respect to "betaT"

#Find clusters

cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data,rank=0)
cluster_finder.find_clusters()

#Retrieve results

results = cluster_finder.results_dict

detection_processor = szifi.detection_processor(results,params_szifi)

catalogue_obs_noit = detection_processor.results.catalogues["catalogue_find_0"]
catalogue_obs_it = detection_processor.results.catalogues["catalogue_find_1"]

#Postprocess detections

#Reimpose threshold

q_th_final = 5.

catalogue_obs_noit = szifi.get_catalogue_q_th(catalogue_obs_noit,q_th_final)
catalogue_obs_it = szifi.get_catalogue_q_th(catalogue_obs_it,q_th_final)

#Merge catalogues of all fields

radius_arcmin = 10. #merging radius in arcmin

catalogue_obs_noit = szifi.merge_detections(catalogue_obs_noit,radius_arcmin=radius_arcmin,return_merge_flag=True,mode="fof")
catalogue_obs_it = szifi.merge_detections(catalogue_obs_it,radius_arcmin=radius_arcmin,return_merge_flag=True,mode="fof")

#Some plots

pl.hist(catalogue_obs_it.catalogue["q_opt"],color="tab:blue",label="Iterative")
pl.hist(catalogue_obs_noit.catalogue["q_opt"],color="tab:orange",label="Non iterative")
pl.legend()
pl.xlabel("Detection SNR")
pl.ylabel("Number of detections")
pl.savefig("detection_histogram.pdf")
pl.show()

pl.scatter(catalogue_obs_noit.catalogue["q_opt"],catalogue_obs_it.catalogue["q_opt"])
x = np.linspace(np.min(catalogue_obs_noit.catalogue["q_opt"]),np.max(catalogue_obs_noit.catalogue["q_opt"]),100)
pl.plot(x,x,color="k")
pl.xlabel("Non-iterative SNR")
pl.ylabel("Iterative SNR")
pl.savefig("detection_itnoit_comparison.pdf")
pl.show()
