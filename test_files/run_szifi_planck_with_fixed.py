import szifi
import numpy as np
import pylab as pl

#Version of run_szifi_planck.py that re-extracts the SNR at the output catalogue,
#illustrating the use of the fixed mode of SZiFi

#Set parameters

params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

params_szifi = szifi.params_szifi_default

#Input data

params_data["field_ids"] = [0,1]
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

#Find clusters

cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data)
cluster_finder.find_clusters()

#Retrieve results

results = cluster_finder.results_dict

detection_processor = szifi.detection_processor(results,params_szifi)

catalogue_obs_noit = detection_processor.results.catalogues["catalogue_find_0"]
catalogue_obs_it = detection_processor.results.catalogues["catalogue_find_1"]

#Re-extract in fixed mode using as input "fixed" catalogue the blind output catalogue

#Define fixed catalogue

data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

data.data["catalogue_input"] = {}
data.data["catalogue_input"][0] = catalogue_obs_it

#Run SZiFi on fixed mode

params_szifi["extraction_mode"] = "fixed"

cluster_finder = szifi.cluster_finder(params_szifi=params_szifi,params_model=params_model,data_file=data)
cluster_finder.find_clusters()

results = cluster_finder.results_dict

detection_processor = szifi.detection_processor(results,params_szifi)

catalogue_obs_it_fixed = detection_processor.results.catalogues["catalogue_fixed_1"]

print(catalogue_obs_it.catalogue["q_opt"])
print(catalogue_obs_it_fixed.catalogue["q_opt"])

#As expected, the SNR measurements of both catalogues coincide
