import numpy as np
import szifi

#Sample programme to compute the master coupling matrix for a given mask

params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

params_data["field_ids"] = [0,1]
params_data["lrange"] = [100,2500]
fac = 4

data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

path = params_szifi["path"]

fields = [0,1]

for field_id in fields:

    mask = data.data["mask_ps"][field_id]
    pix = data.data["pix"][field_id]

    ps = szifi.power_spectrum(pix,mask=mask,cm_compute=True,cm_compute_scratch=True,
    fac=fac,cm_save=True,cm_name=path + "/data/coupling_matrix_" + str(field_id) + ".fits")
