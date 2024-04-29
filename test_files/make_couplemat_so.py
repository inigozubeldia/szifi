import numpy as np
import szifi
#Sample programme to compute the master coupling matrix for a given mask

params_szifi = szifi.params_szifi_so
params_data = szifi.params_data_so
params_model = szifi.params_model_default
bin_fac = 4
lmax1d = 20000
#fields = np.loadtxt('/nvme1/scratch/erosen/data/so_sims/cmb+noise+sz_car_f32/healpix_ids_sosimmask_nside008.txt').astype('int64')
fields = np.arange(273, 274)
params_szifi['powspec_lmax1d'] = lmax1d
params_szifi['powspec_bin_fac'] = bin_fac
path = params_szifi["path"]

for field_id in fields:
    params_data['field_ids'] = [field_id]
    data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

    mask = data.data["mask_ps"][field_id]
    pix = data.data["pix"][field_id]
    
    new_shape = szifi.maps.get_newshape_lmax1d((pix.nx, pix.nx), lmax1d, pix.dx, False)
    print(field_id, new_shape)
    print(data.data['coupling_matrix_name'][field_id])
    ps = szifi.power_spectrum(pix,mask=mask,cm_compute=True,cm_compute_scratch=False,
                              bin_fac=bin_fac, new_shape=new_shape, cm_save=True,cm_name=data.data['coupling_matrix_name'][field_id])
