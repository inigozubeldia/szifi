import numpy as np
from pixell import enmap
import szifi

path_save = "/rds/project/rds-YVo7YUJF2mk/so-cfc/sims/cmb+noise+sz/car_tiles/"
path = "/rds/project/rds-YVo7YUJF2mk/so-cfc/sims/cmb+noise+sz/"

map_names = [
"cmb+noise+sz_093GHz_cmbseed001_beamfwhm_02p2arcmin_white008uK-arcmin_seed094_pixwin.fits",
"cmb+noise+sz_145GHz_cmbseed001_beamfwhm_01p4arcmin_white010uK-arcmin_seed146_pixwin.fits",
"cmb+noise+sz_225GHz_cmbseed001_beamfwhm_01p0arcmin_white022uK-arcmin_seed226_pixwin.fits",
"cmb+noise+sz_278GHz_cmbseed001_beamfwhm_00p9arcmin_white054uK-arcmin_seed279_pixwin.fits",
]

field_ids = ['1_0_1'] #Specify here the tile ids

for i in range(0,len(map_names)):

    print(i)

    input_map = enmap.read_map(path + map_names[i])

    for field_id in field_ids:

        radec = szifi.sphere.get_radec_for_tile('SOCFC_car_tiles/tileDefinitions.yml',field_id)

        expansion_deg = 1.
        tile_map, mask_tile = szifi.maps.get_expanded_map_car(input_map, radec, expansion_deg=expansion_deg)
        
        enmap.write_map(path_save + "tile_map_" + field_id + "_" + str(i),tile_map)

        if i == 0:

            enmap.write_map(path_save + "tile_mask_" + field_id,mask_tile)

    del input_map
    import gc
    gc.collect()