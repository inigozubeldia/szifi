import numpy as np
import healpy as hp
from szifi import maps
from szifi import expt

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

        self.nside_tile = 8
        self.n_tile = hp.nside2npix(self.nside_tile)

        self.nx = 1024 #number of pixels per dimension
        self.l = 14.8  #field size in deg
        self.dx_arcmin = self.l/self.nx*60. #pixel size in arcmin
        self.dx = self.dx_arcmin/180./60.*np.pi
        self.pix = maps.pixel(self.nx,self.dx)

        self.data["nside_tile"] = self.nside_tile

        for field_id in field_ids:

            #Pixelation

            self.data["nx"][field_id] = 1024
            self.data["ny"][field_id] = 1024
            self.data["dx_arcmin"][field_id] = self.dx_arcmin
            self.data["dy_arcmin"][field_id] = self.dx_arcmin

            self.data["pix"][field_id] = self.pix

            #Fields

            [tmap] = np.load(path + "planck_maps/planck_field_" + str(field_id) + "_tmap.npy")
            tmap[:,:,4] = tmap[:,:,4]/58.04
            tmap[:,:,5] = tmap[:,:,5]/2.27
            tmap = tmap*1e6

            self.data["t_obs"][field_id] = tmap
            self.data["t_noi"][field_id] = tmap

            #Masks

            buffer_arcmin = 10. #usually around twice the beam

            [mask_galaxy,mask_point,mask_tile] = np.load(path + "planck_maps/planck_field_" + str(field_id) + "_mask.npy")

            mask_ps = maps.get_apodised_mask(self.pix,mask_galaxy,apotype="Smooth",aposcale=0.2)
            mask_ps = maps.get_apodised_mask(self.pix,mask_galaxy,apotype="Smooth",aposcale=0.2)

            mask_peak_finding_no_tile = mask_galaxy*mask_point
            mask_select_no_tile = maps.get_buffered_mask(self.pix,mask_peak_finding_no_tile,buffer_arcmin,type="fft")
            mask_peak_finding = mask_peak_finding_no_tile*mask_tile
            mask_select = mask_select_no_tile*mask_tile
            mask_select = maps.get_fsky_criterion_mask(self.pix,mask_select,self.nside_tile,criterion=params_szifi["min_ftile"])

            if params_szifi["tilemask_mode"] == "catalogue_buffer":
                mask_select_buffer = mask_select_no_tile * \
                                        maps.get_buffer_region(self.pix, mask_tile, params_szifi["tilemask_buffer_arcmin"])
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
            self.data["mask_tile"][field_id] = mask_tile
            #Coupling matrix

            if np.array_equal(mask_ps, maps.get_apodised_mask(self.pix,np.ones((self.nx,self.nx)),
            apotype="Smooth",aposcale=0.2)):

                    cm_name = path + "coupling_matrices_planck/apod_smooth_1024.fits"

            else:

                cm_name = path + "coupling_matrices_planck/apod_smooth_" + str(field_id) + ".fits"

            self.data["coupling_matrix_name"][field_id] = cm_name

        #Experiment specifications

        self.data["experiment"] = expt.experiment(experiment_name="Planck_real",params_szifi=params_szifi)
