import numpy as np
import szifi
import os
i_min = 0
i_max = 768
merge_length_obs = 10.
id_mode = "close+highest_q"
path = "/rds-d4/user/iz221/hpc-work/catalogues_szifi_planck/"
find_label = False
cib_random_labels = ["_cib_random","_cib_correlated"]
for k in range(0,len(cib_random_labels)): #For simulated data
#for k in range(0,1): #For real data
    cib_random_label = cib_random_labels[k] #just set to "" for real data
    sim_label = "_sim_" # "_sim" #just set to "" for real data
    if find_label == True:
        prename = "planck_it_find_apodold_planckcibparams" + sim_label + cib_random_label
        suffixes = [
        "mmf_6",
        "mmf_5",
        "mmf_4",
        "cmmf_6",
        "cmmf_5",
        "cmmf_4",
        "cmmf_beta_6",
        "cmmf_betaT_6",
        "cmmf_beta_5",
        "cmmf_betaT_5",
        ]
    elif find_label == False:
        prename = "planck_it_fixedmean_apodold_planckcibparams" + sim_label + cib_random_label
        print("name",prename)
        suffixes = [
        "mmf_6",
        "mmf_5",
        "mmf_4",
        "cmmf_6",
        "cmmf_5",
        "cmmf_4",
        "cmmf_beta_6",
        "cmmf_betaT_6",
        "cmmf_beta_5",
        "cmmf_betaT_5",
        "mf_1",
        "mf_2",
        "mf_3",
        "mf_4",
        "mf_5",
        "mf_6",
        ]
    for i in range(0,len(suffixes)):
        name_full = prename + "_" + suffixes[i]
        print(name_full)
        results_dict = {}
        n_total = 0
        for j in range(i_min,i_max):
            name = "/rds-d4/user/iz221/hpc-work/catalogues_szifi_planck/" + name_full + "_" + str(j) + ".npy"
            if os.path.isfile(name) == True:
                rd = np.load(name,allow_pickle=True)[()]
                catalogue_keys = rd[j].catalogues.keys()
                print("Catalogue keys",catalogue_keys)
                if  "catalogue_fixed_1" in catalogue_keys:
                    n_total = n_total + len(rd[j].catalogues["catalogue_fixed_1"].catalogue["q_opt"])
                    #print(j,n_total)
                results_dict.update(rd)
        params_szifi = {"theta_500_vec_arcmin":np.exp(np.linspace(np.log(0.5),np.log(15.),15)),
        "iterative":True}
        detection_processor = szifi.detection_processor(results_dict,params_szifi)
        if find_label == True:
            catalogue_obs_noit = detection_processor.results.catalogues["catalogue_find_0"]
            catalogue_obs_it = detection_processor.results.catalogues["catalogue_find_1"]
            print("Pixel id max",np.max(catalogue_obs_it.catalogue["pixel_ids"]))
            print(catalogue_obs_it.catalogue["q_opt"])
            q_th_final = 5.
            catalogue_obs_noit = szifi.get_catalogue_q_th(catalogue_obs_noit,q_th_final)
            catalogue_obs_it = szifi.get_catalogue_q_th(catalogue_obs_it,q_th_final)
            print(len(catalogue_obs_it.catalogue["pixel_ids"]))
            print("Merging catalogue obs")
            radius_arcmin = 2. #merging radius in arcmin
            catalogue_obs_noit = szifi.merge_detections(catalogue_obs_noit,radius_arcmin=merge_length_obs,return_merge_flag=True,mode="fof")
            catalogue_obs_it = szifi.merge_detections(catalogue_obs_it,radius_arcmin=merge_length_obs,return_merge_flag=True,mode="fof")
            print("Catalogue obs merged")
        elif find_label == False:
            print(detection_processor.results.catalogues.keys())
            catalogue_obs_it = detection_processor.results.catalogues["catalogue_fixed_1"]
            catalogue_obs_noit = catalogue_obs_it
            print("Pixel id max",np.max(catalogue_obs_it.catalogue["pixel_ids"]))
            print("q min",np.min(catalogue_obs_it.catalogue["q_opt"]))
            print("q max",np.max(catalogue_obs_it.catalogue["q_opt"]))
            print("n",len(catalogue_obs_it.catalogue["q_opt"]))
        full_name = path + prename + "_" + suffixes[i] + "_processed.npy"
        metadata = {"name":full_name,
        "merge_legth_obs":merge_length_obs}
        print(full_name)
        np.save(full_name,(catalogue_obs_noit,catalogue_obs_it),allow_pickle=True)
