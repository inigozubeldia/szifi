import numpy as np
import healpy as hp
import szifi

#Sample programme to get flat-sky square cutouts from HEALpix map

params_szifi = szifi.params_szifi_default
params_data = szifi.params_data_default
params_model = szifi.params_model_default

data = szifi.input_data(params_szifi=params_szifi,params_data=params_data)

nx = data.nx
l = data.l

n_tiles = data.n_tile
nside_tile = data.nside_tile
tiles = np.arange(0,n_tiles)

map = np.load("your_healpy_map.npy")

for i in range(0,n_tiles):

    lon,lat = hp.pix2ang(nside_tile,i,lonlat=True)
    cutout = szifi.get_cutout(map,[lon,lat],nx,l)

    np.save("your_cutout_path/cutout_" + str(i) + ".npy",cutout)
