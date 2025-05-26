import numpy as np
from szifi import maps
from szifi import expt
from pixell import enmap
import szifi
import os

class input_data_survey:

    def __init__(self,params_szifi=None,params_data=None):

        path = params_szifi["path_data"]
        field_ids = params_data["field_ids"]

        self.data = {}
        self.data["params_data"] = params_data

        self.data["nx"] = {} #Field number of pixels along each dimension
        self.data["ny"] = {} #Field number of pixels along each dimension
        self.data["dx_arcmin"] = {} #Field pixel size in arcmin
        self.data["dy_arcmin"] = {} #Field pixel size in arcmin
        self.data["pix"] = {}

        self.data["t_obs"] = {} #Temperature maps in muK, nx x nx x n_freq
        self.data["t_noi"] = {} #Temperature maps to be used for noise estimation in muK, nx x nx x n_freq (in genereal, same as "t_obs", unless the maps minus the tSZ signal is known)

        self.data["mask_map"] = {}
        self.data["mask_point"] = {}
        self.data["mask_select"] = {}
        self.data["mask_select_no_tile"] = {}
        self.data["mask_select_buffer"] = {}
        self.data["mask_ps"] = {}
        self.data["mask_peak_finding"] = {}
        self.data["mask_peak_finding_no_tile"] = {}
        self.data["mask_tile"] = {}

        self.data["coupling_matrix_name"] = {}

        for field_id in field_ids:

            #Load maps

            mask_tile = enmap.read_map(path + "tile_mask_" + field_id)

            input_maps_list = []

            for freq_id in [0,1,2,3]:

                input_maps_list.append(enmap.read_map(path + "tile_map_" + field_id + "_" + str(freq_id)))
            
            input_maps = np.stack(input_maps_list)
            wcs = input_maps_list[0].wcs 
            tile_map = enmap.enmap(input_maps,wcs)

            #Pixelation

            print(tile_map.shape)
            nx = tile_map.shape[1]
            ny = tile_map.shape[2]
            dx,dy = tile_map.extent()
            dx = dx/nx
            dy = dy/ny

            pix = maps.pixel(nx,dx,ny=ny,dy=dy)

            self.data["nx"][field_id] = nx
            self.data["ny"][field_id] = ny
            self.data["dx_arcmin"][field_id] = dx*180.*60./np.pi
            self.data["dy_arcmin"][field_id] = dy*180.*60./np.pi
            self.data["pix"][field_id] = pix

            tile_map = enmap.enmap(np.moveaxis(tile_map, 0, -1),tile_map.wcs)

            self.data["t_obs"][field_id] = tile_map
            self.data["t_noi"][field_id] = tile_map

            #Masks

            buffer_arcmin = 2. #usually around twice the beam
            mask_galaxy = np.ones_like(mask_tile)
            mask_point = np.ones_like(mask_galaxy)

            mask_ps = maps.get_apodised_mask(pix,mask_galaxy,apotype="Smooth",aposcale=0.01)
            mask_peak_finding_no_tile = mask_galaxy*mask_point
            mask_select_no_tile = maps.get_buffered_mask(pix,mask_peak_finding_no_tile,buffer_arcmin,type="fft",tile_type=params_szifi["tile_type"],wcs=wcs)
            mask_peak_finding = mask_peak_finding_no_tile*mask_tile
            mask_select = mask_select_no_tile*mask_tile
            mask_select_no_tile = maps.get_fsky_criterion_mask(pix,mask_select_no_tile,0,criterion=params_szifi["min_ftile"],tile_type=params_szifi["tile_type"])

            if np.all(np.abs(mask_select_no_tile) < 1e-5) is True:

                mask_select = np.zeros(mask_select.shape)

            if params_szifi["tilemask_mode"] == "catalogue_buffer":
                mask_select_buffer = mask_select_no_tile * \
                                        maps.get_buffer_region(pix,mask_tile,params_szifi["tilemask_buffer_arcmin"])
            else:
                mask_select_buffer = 0

            self.data["mask_point"][field_id] = mask_point
            self.data["mask_select"][field_id] = mask_select
            self.data["mask_select_no_tile"][field_id] = mask_select_no_tile
            self.data["mask_select_buffer"][field_id] = mask_select_buffer
            self.data["mask_map"][field_id] = mask_ps
            self.data["mask_ps"][field_id] = mask_ps
            self.data["mask_peak_finding_no_tile"][field_id] = mask_peak_finding_no_tile
            self.data["mask_peak_finding"][field_id] = mask_peak_finding

            #Coupling matrix

            powspec_lmax1d = params_szifi['powspec_lmax1d']
            if powspec_lmax1d is None: lmax_tag = ""
            else: lmax_tag = f"_lmax{powspec_lmax1d:05d}"
            bin_fac = params_szifi['powspec_bin_fac']

            cm_name = f"/rds/project/rds-YVo7YUJF2mk/so-cfc/szifi/coupling_matrices_so/apod_smooth_field_half_{field_id}_bin{bin_fac:02d}{lmax_tag}.fits"

            self.data["coupling_matrix_name"][field_id] = cm_name


        #Experiment specifications

        self.data["experiment"] = expt.experiment(experiment_name="SObaseline_simple",params_szifi=params_szifi)


