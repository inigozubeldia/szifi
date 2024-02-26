import numpy as np
import szifi
#Sample programme to compute the master coupling matrix for a given mask

params_szifi = szifi.params_szifi_so
params_data = szifi.params_data_so
params_model = szifi.params_model_default
bin_fac = 4
lmax1d = 10000
fields = [208, 209]
params_data['powspec_lmax1d'] = lmax1d
params_data['powspec_bin_fac'] = bin_fac
params_data['field_ids'] = fields
data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)
path = params_szifi["path"]

for field_id in fields:

    mask = data.data["mask_ps"][field_id]
    pix = data.data["pix"][field_id]
    
    new_shape = szifi.maps.get_newshape_lmax1d((pix.nx, pix.nx), lmax1d, pix.dx, False)
    print(new_shape)
    ps = szifi.power_spectrum(pix,mask=mask,cm_compute=True,cm_compute_scratch=True,
                              bin_fac=bin_fac, new_shape=new_shape, cm_save=True,cm_name=data.data['coupling_matrix_name'][field_id])
