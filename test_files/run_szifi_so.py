import szifi
import numpy as np
import pylab as pl

#Sample programme that runs SZiFi on Planck data for two sky tiles.
#It assumes that the mask coupling matrix
#has already been calculated (see get_coupling_matrix.py)

#Set parameters

params_szifi = szifi.params_szifi_so
params_data = szifi.params_data_so
params_model = szifi.params_model_default

lmax1d = 10000
bin_fac=4
params_szifi["powspec_lmax1d"] = lmax1d
params_szifi["powspec_bin_fac"] = bin_fac
#Input data
# params_szifi['compute_coupling_matrix'] = True
# params_szifi['save_coupling_matrix'] = True
params_data["field_ids"] = [208]
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

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
