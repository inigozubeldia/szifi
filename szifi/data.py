import numpy as np
import healpy as hp
from .params import *
from .maps import *

class input_data:

    def __init__(self,params_szifi=params_szifi_default,params_data=params_data_default):

        path = params_szifi["path_data"]
        field_ids = params_data["field_ids"]

        self.data = {}
        self.data["params_data"] = params_data

        self.data["nx"] = {} #Field number of pixels along each dimension
        self.data["dx_arcmin"] = {} #Field pixel size in arcmin
        self.data["t_obs"] = {} #Temperature maps in muK, nx x nx x n_freq
        self.data["t_noi"] = {} #Temperature maps to be used for noise estimation in muK, nx x nx x n_freq (in genereal, same as "t_obs", unless the maps minus the tSZ signal is known)

        self.data["mask_map"] = {}
        self.data["mask_point"] = {}
        self.data["mask_select"] = {}
        self.data["mask_select_no_tile"] = {}
        self.data["mask_ps"] = {}
        self.data["mask_peak_finding"] = {}
        self.data["mask_peak_finding_no_tile"] = {}
        self.data["pix"] = {}

        self.data["coupling_matrix_name"] = {}

        if params_data["data_set"] == "Planck_real":

            self.nside_tile = 8
            self.n_tile = hp.nside2npix(self.nside_tile)

            self.nx = 1024 #number of pixels per dimension
            self.l = 14.8  #field size in deg
            self.dx_arcmin = self.l/self.nx*60. #pixel size in arcmin
            self.dx = self.dx_arcmin/180./60.*np.pi
            self.pix = pixel(self.nx,self.dx)

            self.data["nside_tile"] = self.nside_tile

            for field_id in field_ids:

                #Pixelation

                self.data["nx"][field_id] = 1024
                self.data["dx_arcmin"][field_id] = self.dx_arcmin
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

                mask_ps = get_apodised_mask(self.pix,mask_galaxy,apotype="Smooth",aposcale=0.2)
                mask_ps = get_apodised_mask(self.pix,mask_galaxy,apotype="Smooth",aposcale=0.2)

                mask_peak_finding_no_tile = mask_galaxy*mask_point
                mask_select_no_tile = get_buffered_mask(self.pix,mask_peak_finding_no_tile,buffer_arcmin,type="fft")
                mask_peak_finding = mask_peak_finding_no_tile*mask_tile
                mask_select = mask_select_no_tile*mask_tile
                mask_select = get_fsky_criterion_mask(self.pix,mask_select,self.nside_tile,criterion=params_szifi["min_ftile"])

                self.data["mask_point"][field_id] = mask_point
                self.data["mask_select"][field_id] = mask_select
                self.data["mask_select_no_tile"][field_id] = mask_select_no_tile
                self.data["mask_map"][field_id] = mask_ps
                self.data["mask_ps"][field_id] = mask_ps
                self.data["mask_peak_finding_no_tile"][field_id] = mask_peak_finding_no_tile
                self.data["mask_peak_finding"][field_id] = mask_peak_finding

                #Coupling matrix

                if np.array_equal(mask_ps,get_apodised_mask(self.pix,np.ones((self.nx,self.nx)),
                apotype="Smooth",aposcale=0.2)):

                     cm_name = path + "coupling_matrices_planck/apod_smooth_1024.fits"

                else:

                    cm_name = path + "coupling_matrices_planck/apod_smooth_" + str(field_id) + ".fits"

                self.data["coupling_matrix_name"][field_id] = cm_name

            #Experiment specifications

            self.data["experiment"] = experiment(experiment_name="Planck_real",params_szifi=params_szifi)


        elif params_data["data_set"] == "Planck_websky":

            for field_id in field_ids:

                #Fields

                # self.data["mask_point"][field_id] = mask_point
                # self.data["mask_select"][field_id] = mask_select
                # self.data["mask_select_no_tile"][field_id] = mask_select_no_tile
                # self.data["mask_map"][field_id] = mask_ps
                # self.data["mask_ps"][field_id] = mask_ps

                #Coupling matrix

                self.data["coupling_matrix_name"][field_id] = path + "data/apod_smooth_1024.fits"


            #Experiment specifications

            #Coupling matrix


class catalogue_data:

    def __init__(self,name,type=None,params_szifi=params_szifi_default):

        path = params_szifi["path"] + "data/"

        if name == "Planck_SZ":

            fit_union = fits.open('HFI_PCCS_SZ-union_R2.08.fits')
            fit_mmf3 = fits.open('HFI_PCCS_SZ-MMF3_R2.08.fits')

            data_union = fit_union[1].data
            data_mmf3 = fit_mmf3[1].data

            if type == "mmf3":

                indices_mmf3 = np.where(data_mmf3["SNR"] > threshold)
                indices_union = data_mmf3["INDEX"][indices_mmf3]-1

            elif type == "cosmo":

                indices_mmf3 = []
                indices_union = []

                for i in range(0,len(data_mmf3["SNR"])):

                    if (data_mmf3["SNR"][i] > threshold) and (data_union["COSMO"][data_mmf3["INDEX"][i]-1] == True):

                        indices_union.append(data_mmf3["INDEX"][i]-1)
                        indices_mmf3.append(i)

            self.catalogue = cluster_catalogue()

            self.catalogue.catalogue["q_opt"] = data_mmf3["SNR"][indices_mmf3]
            self.catalogue.catalogue["lon"] = data_union["GLON"][indices_union]
            self.catalogue.catalogue["lat"] = data_union["GLAT"][indices_union]
            self.catalogue.catalogue["z"] = data_union["REDSHIFT"][indices_union]
            self.catalogue.catalogue["M_SZ"] = data_union["MSZ"][indices_union]

        elif name == "Planck_gcc":

            cat_fits = fits.open(path + "HFI_PCCS_GCC_R2.02.fits")
            data = cat_fits[1].data

            self.catalogue = cluster_catalogue()

            self.catalogue.catalogue["lon"] = data["GLON"]
            self.catalogue.catalogue["lat"] = data["GLAT"]
            self.catalogue.catalogue["flux_quality"] = data["FLUX_QUALITY"]

        elif name == "Planck_cs":

            names = [path + "COM_PCCS_857_R2.01.fits",
            path + "COM_PCCS_545_R2.01.fits"]

            lon = np.empty(0)
            lat = np.empty(0)

            for i in range(0,len(names)):

                cat_fits = fits.open(names[i])
                data = cat_fits[1].data

                lon = np.append(lon,data["GLON"])
                lat = np.append(lat,data["GLAT"])

            self.catalogue = cluster_catalogue()

            self.catalogue.catalogue["lon"] = lon
            self.catalogue.catalogue["lat"] = lat

        elif name == "clusterdb":

            cat_fits = fits.open(path + 'masterclusters.fits')
            data = cat_fits[1].data

            print(data.dtype.names)

            print(data["name"])
            print(data["redshift"])
            print(data["mcsz.name"])

            self.ra = data["Ra"]
            self.dec = data["Dec"]

            gc = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree, frame='icrs')
            coords = gc.transform_to('galactic')

            self.lon = np.zeros(len(self.ra))
            self.lat = np.zeros(len(self.ra))

            for i in range(0,len(self.ra)):

                self.lon[i] = coords[i].l.value
                self.lat[i] = coords[i].b.value

            self.catalogue = cluster_catalogue()

            self.catalogue.catalogue["lon"] = self.lon
            self.catalogue.catalogue["lat"] = self.lat

            self.catalogue.catalogue["lon_m2c"] = self.lon
            self.catalogue.catalogue["lat_m2c"] = self.lat
            self.catalogue.catalogue["redshift_m2c"] = data["Redshift"]
            self.catalogue.catalogue["name_m2c"] = data["name"]
            self.catalogue.catalogue["m_yx_m2c"] = data["m_yx"]
            self.catalogue.catalogue["mcsz.name"] = data["mcsz.name"]
            self.catalogue.catalogue["mcxc2021.name_mcxc"] = data["mcxc2021.name_mcxc"]
            self.catalogue.catalogue["comprass.name"] = data["comprass.name"]
            self.catalogue.catalogue["Dec_m2c"] = data["Dec"]
            self.catalogue.catalogue["Ra_m2c"] = data["Ra"]

            self.keys_unique = ["lon_m2c","lat_m2c","redshift_m2c","name_m2c","m_yx_m2c","mcsz.name",
            "mcxc2021.name_mcxc","comprass.name","Ra_m2c","Dec_m2c"]

        elif name == "ACT_dr5":

            cat_fits = fits.open(path + "DR5_cluster-catalog_v1.1.fits")
            data = cat_fits[1].data

            self.ra = data["RADeg"]
            self.dec = data["decDeg"]
            self.redshift = data["redshift"]

            print(data.dtype.names)

            keys_act = list(data.dtype.names)
            keys_unique = []

            for i in range(0,len(keys_act)):

                keys_unique.append(keys_act[i] + "_act")

            self.keys_unique = keys_unique

            gc = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree, frame='icrs')
            coords = gc.transform_to('galactic')

            self.lon = np.zeros(len(self.ra))
            self.lat = np.zeros(len(self.ra))

            for i in range(0,len(self.ra)):

                self.lon[i] = coords[i].l.value
                self.lat[i] = coords[i].b.value

            self.catalogue = cluster_catalogue()

            self.catalogue.catalogue["lon"] = self.lon
            self.catalogue.catalogue["lat"] = self.lat

            for i in range(0,len(keys_unique)):

                self.catalogue.catalogue[keys_unique[i]] = data[keys_act[i]]

        elif name == "PSZ_MCMF":

            cat_fits = fits.open(path + "PSZ-SN3_cluster_catalogue.fits")
            data = cat_fits[1].data

            self.ra = data["RA_{Planck}"]
            self.dec = data["DEC_{Planck}"]
            self.snr_planck = data["S/N"]
            self.z = data["z"]
            self.name = data["ID"]

            gc = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree, frame='icrs')
            coords = gc.transform_to('galactic')

            self.lon = np.zeros(len(self.ra))
            self.lat = np.zeros(len(self.ra))

            for i in range(0,len(self.ra)):

                self.lon[i] = coords[i].l.value
                self.lat[i] = coords[i].b.value

            self.catalogue = cluster_catalogue()

            self.catalogue.catalogue["lon"] = self.lon
            self.catalogue.catalogue["lat"] = self.lat
            self.catalogue.catalogue["name"] = self.name
            self.catalogue.catalogue["q_opt"] = self.snr_planck
