import numpy as np
import pylab as pl
import healpy as hp
import expt
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import sphere
import cat
from matplotlib.patches import Circle
import astropy.cosmology as cp
import maps
import mmf
import os

class cosmology_websky:

    def __init__(self):

        self.Ob0 = 0.049
        self.Oc0 = 0.261
        self.Om0 = self.Ob0 + self.Oc0
        self.h      = 0.68
        self.ns     = 0.965
        self.sigma8 = 0.81
        self.cosmology = cp.FlatLambdaCDM(Om0=self.Om0,H0=self.h*100.,Ob0=self.Ob0)

class catalogue_websky_master:

    def __init__(self,mode="websky",path="/Users/user/Desktop/"):

        if mode == "websky":

            catalogue = np.load(path + "data/websky/halos_reduced.npy")
            catalogue_mass_converted = np.load(path + "data/websky/halos_converted_mass.npy")
            y0 = np.load(path + "data/websky/halos_y0.npy")

            self.x = catalogue[:,0]
            self.y = catalogue[:,1]
            self.z = catalogue[:,2]
            self.M200m = catalogue[:,3]
            self.redshift = catalogue[:,4]
            self.chi = catalogue[:,5]
            self.vrad = catalogue[:,6]
            self.r_s = catalogue[:,7]
            self.R200m = catalogue[:,8]/(200.)**(1./3.)
            self.theta_200m = catalogue[:,9]/(200.)**(1./3.)
            self.lon = catalogue[:,10]
            self.lat = catalogue[:,11]
            self.M500c = catalogue_mass_converted[:,0]
            self.R500c = catalogue_mass_converted[:,1]/(200.)**(1./3.)
            self.theta_500c = self.R500c/self.chi
            self.y0 = y0

        elif mode == "injected":

            catalogue = np.load(path + "data/websky/halos_injected_2.npy")
            self.x = catalogue[:,0]
            self.y = catalogue[:,1]
            self.z = catalogue[:,2]
            self.M200m = catalogue[:,3]
            self.redshift = catalogue[:,4]
            self.chi = catalogue[:,5]
            self.vrad = catalogue[:,6]
            self.r_s = catalogue[:,7]
            self.R200m = catalogue[:,8]
            self.theta_200m = catalogue[:,9]
            self.lon = catalogue[:,10]
            self.lat = catalogue[:,11]
            self.M500c = catalogue[:,12]
            self.R500c = catalogue[:,13]
            self.theta_500c = catalogue[:,14]
            self.y0 = catalogue[:,15]

    def select_threshold(self,select="mass",th=2.,m_min=1e14):

        if select == "mass":

            indices = np.where((self.M200m > m_min))

        elif select == "q":

            z_vec,M_500_vec = np.load("threshold_curve_" + str(th) + ".npy")
            th = self.M200m-np.interp(self.redshift,z_vec,M_500_vec*1e15)
            indices = np.where((th > 0.))

        return catalogue_websky(
        theta_200m=self.theta_200m[indices],lat=self.lat[indices],lon=self.lon[indices],M200m=self.M200m[indices],
        redshift=self.redshift[indices],R200m=self.R200m[indices],M500c=self.M500c[indices],R500c=self.R500c[indices],
        theta_500c=self.theta_500c[indices],y0=self.y0)

    def select_tile(self,i,pix,m_min=1e14,type="field",nside=8,select="mass",th=2.):

        theta_x,theta_y = sphere.get_xy(i,self.lon,self.lat,nside)

        if select == "mass":

            indices1 = np.where((self.M200m > m_min))

        elif select == "q":

            z_vec,M_500_vec = np.load("threshold_curve_" + str(th) + ".npy")
            th = self.M200m-np.interp(self.redshift,z_vec,M_500_vec*1e15)
            indices1 = np.where((th > 0.))

        theta_x = theta_x[indices1]
        theta_y = theta_y[indices1]
        theta_200m = self.theta_200m[indices1]
        M200m = self.M200m[indices1]
        lon = self.lon[indices1]
        lat = self.lat[indices1]
        redshift = self.redshift[indices1]
        R200m = self.R200m[indices1]
        M500c = self.M500c[indices1]
        R500c = self.R500c[indices1]
        theta_500c = self.theta_500c[indices1]
        y0 = self.y0[indices1]

        if type == "field":

            lx = pix.nx*pix.dx
            indices = np.where((theta_x < lx*0.5) & (theta_x > -lx*0.5) & (theta_y < lx*0.5) & (theta_y > -lx*0.5))

        elif type == "tile":

            pix_vec = hp.pixelfunc.ang2pix(nside,lon,lat,lonlat=True)
            indices = np.where((pix_vec == i))

        theta_x = theta_x + pix.nx*pix.dx*0.5
        theta_y = theta_y + pix.nx*pix.dx*0.5

        return catalogue_websky(theta_x=theta_x[indices],theta_y=theta_y[indices],
        theta_200m=theta_200m[indices],lat=lat[indices],lon=lon[indices],M200m=M200m[indices],
        redshift=redshift[indices],R200m=R200m[indices],M500c=M500c[indices],R500c=R500c[indices],
        theta_500c=theta_500c[indices],y0=y0[indices])

    def get_catalogue_measurement(self):

        negative = -np.ones(len(self.lat))*0.5

        return cat.cluster_catalogue(q_opt=negative
        ,y0=self.y0,theta_500=self.theta_500c*180.*60./np.pi,theta_x=self.theta_x
        ,theta_y=self.theta_y,
        pixel_ids=negative,lat=self.lat,
        lon=self.lon,m_500=self.M500c,z=self.redshift)

class catalogue_websky:

    def __init__(self,theta_x=np.empty(0),theta_y=np.empty(0),theta_200m=np.empty(0),
    lat=np.empty(0),lon=np.empty(0),theta_500c=np.empty(0),M200m=np.empty(0),M500c=np.empty(0),
    R200m=np.empty(0),R500c=np.empty(0),redshift=np.empty(0),y0=np.empty(0)):

        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_200m = theta_200m
        self.theta_500c = theta_500c
        self.lat = lat
        self.lon = lon
        self.M200m = M200m
        self.M500c = M500c
        self.R200m = R200m
        self.R500c = R500c
        self.redshift = redshift
        self.theta_500 = theta_500c
        self.y0 = y0

    def get_catalogue_measurement(self):

        negative = -np.ones(len(self.lat))*0.5

        cat_return = cat.cluster_catalogue()
        cat_return.catalogue["q_opt"] = negative
        cat_return.catalogue["y0"] = self.y0
        cat_return.catalogue["theta_500"] = self.theta_500c*180.*60./np.pi
        cat_return.catalogue["theta_x"] = self.theta_x
        cat_return.catalogue["theta_y"] = self.theta_y
        cat_return.catalogue["pixel_ids"] = negative
        cat_return.catalogue["lat"] = self.lat
        cat_return.catalogue["lon"] = self.lon
        cat_return.catalogue["m_500"] = self.M500c
        cat_return.catalogue["z"] = self.redshift

        return cat_return

class websky_maps:

    def __init__(self):

        self.exp = expt.websky_specs()

    def get_tsz(self,i):

        return hp.read_map("/Users/user/Desktop/data/websky/tsz_" + str(self.exp.nu_eff_GHz[i])+ "_2048.fits",field=0)


#Produces a set of maps ALL IN muK

def preprocess_websky():

    exp = expt.websky_specs()

    #tSZ

    y_map = hp.read_map('/Users/user/Desktop/data/websky/tsz.fits',field=0)
    y_map = hp.pixelfunc.ud_grade(y_map,2048)

    #hp.mollview(y_map)
    #pl.show()

    for i in range(0,6):

        print(i)

        y_map_conv = hp.sphtfunc.smoothing(y_map,beam_window=expt.planck_specs().get_beam(i,ptf=False)[1])
        t_map = y_map_conv*exp.y2muK[i]

        #hp.mollview(t_map)
        #pl.show()

        hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/tsz_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",t_map)

    """
    #kSZ

    ks_map = hp.read_map('/Users/user/Desktop/data/websky/ksz.fits',field=0) + hp.read_map('/Users/user/Desktop/data/websky/ksz_patchy.fits',field=0)
    ks_map = hp.pixelfunc.ud_grade(ks_map,2048)

    hp.mollview(ks_map)
    pl.show()

    for i in range(0,6):

        print(i)

        t_map = hp.sphtfunc.smoothing(ks_map,beam_window=expt.planck_specs().get_beam(i,ptf=False)[1])

        #hp.mollview(t_map)
        #pl.show()

        hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/ksz_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",t_map)



    #CIB


    for i in range(0,6):

        print(i)

        cib_map = hp.read_map("/Users/user/Desktop/data/websky/cib_nu0" + str(exp.nu_eff_GHz[i]) + ".fits",field=0)

        std = np.std(cib_map)
        indices = np.where(cib_map > 3.*std)[0]
        cib_map[indices] = 0.

        cib_map = hp.pixelfunc.ud_grade(cib_map,2048)
        cib_map = cib_map*exp.MJysr2muK[i]
        t_map = hp.sphtfunc.smoothing(cib_map,beam_window=expt.planck_specs().get_beam(i,ptf=False)[1])

        hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/cib_masked_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",t_map)




    #CMB

    cmb_map = hp.read_map('/Users/user/Desktop/data/websky/cmb.fits',field=0)
    cmb_map = hp.pixelfunc.ud_grade(cmb_map,2048)

    #hp.mollview(ks_map)
    #pl.show()

    for i in range(0,6):

        print(i)

        t_map = hp.sphtfunc.smoothing(cmb_map,beam_window=expt.planck_specs().get_beam(i,ptf=False)[1])

        hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/cmb_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",t_map)

    """

    #tSZ injected

    """

    y_map = hp.read_map('/Users/user/Desktop/data/websky/tsz_injected_2048.fits',field=0)

    hp.mollview(y_map)
    pl.show()

    for i in range(0,6):

        print(i)

        y_map_conv = hp.sphtfunc.smoothing(y_map,beam_window=expt.planck_specs().get_beam(i,ptf=False)[1])
        t_map = y_map_conv*exp.y2muK[i]

        #hp.mollview(t_map)
        #pl.show()

        hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/tsz_injected_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",t_map)

    """

#All in muK

def get_galactic_foregrounds():

    exp = expt.websky_specs()

    dust = pysm3.Sky(nside=2048, preset_strings=["d1"])
    synchro = pysm3.Sky(nside=2048, preset_strings=["s1"])

    for i in range(0,6):

        dust_freq = dust.get_emission(exp.nu_eff_GHz[i]*u.GHz)
        dust_freq = dust_freq.to(u.uK_CMB,equivalencies=u.cmb_equivalencies(exp.nu_eff_GHz[i]*u.GHz))
        dust_freq_t = dust_freq[0]
        dust_freq_t = hp.sphtfunc.smoothing(dust_freq_t,beam_window=expt.planck_specs().get_beam(i,ptf=False)[1])

        hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/dust_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",dust_freq_t)
        #hp.mollview(dust_freq_t,max=1e2)
        #pl.show()

        synchro_freq = synchro.get_emission(exp.nu_eff_GHz[i]*u.GHz)
        synchro_freq = synchro_freq.to(u.uK_CMB,equivalencies=u.cmb_equivalencies(exp.nu_eff_GHz[i]*u.GHz))
        synchro_freq_t = synchro_freq[0]
        synchro_freq_t = hp.sphtfunc.smoothing(synchro_freq_t,beam_window=expt.planck_specs().get_beam(i,ptf=False)[1])
        hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/synchro_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",synchro_freq_t)

        #hp.mollview(synchro_freq_t,max=1e2)
        #pl.show()

#All in muK

def get_noise_maps(ptf_convolution=True):

    exp = expt.websky_specs()
    nside = 2048

    for i in range(0,6):

        print(i)
        if ptf_convolution == True:

            noise_map = sphere.get_white_noise_map_convolved_hp(nside,exp.noise_levels[i])
            hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/noise_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",noise_map)

        elif ptf_convolution == False:

            noise_map = sphere.get_white_noise_map_hp(nside,exp.noise_levels[i])
            hp.fitsfunc.write_map("/Users/user/Desktop/data/websky/noise_no_ptf_" + str(exp.nu_eff_GHz[i])+ "_2048.fits",noise_map)

class cutouts_websky:

    def __init__(self,path="/Users/user/Desktop/",inpaint_flag=False,mode="websky"):

        self.name = path + "maps/websky_maps/t_maps/"
        self.name_mask = path + "maps/planck_maps/planck_field_"
        self.mask_cib = False
        self.sigma_mask = 0.
        self.inpaint_flag = inpaint_flag
        self.nx = 1024
        self.l = 14.8  #in deg
        self.dx = self.l/self.nx/180.*np.pi
        self.pix = maps.pixel(self.nx,self.dx)
        self.nside_tile = 8
        self.n_tile = hp.nside2npix(self.nside_tile)
        self.mode = mode
        self.path = path

    def get_cutout_i(self,i):

        cutout_i = cutout_websky(i,name=self.name,name_mask=self.name_mask,
        mask_cib=self.mask_cib,sigma_mask=self.sigma_mask,inpaint_flag=self.inpaint_flag,mode=self.mode,path=self.path)

        return cutout_i

    def get_cutout_mean(self,i_min,i_max):

        dust = np.zeros((self.nx,self.nx,6))
        synchro = np.zeros((self.nx,self.nx,6))
        cib = np.zeros((self.nx,self.nx,6))
        ksz = np.zeros((self.nx,self.nx,6))
        noise = np.zeros((self.nx,self.nx,6))
        cmb = np.zeros((self.nx,self.nx,6))
        cib_masked = np.zeros((self.nx,self.nx,6))
        noise_no_ptf = np.zeros((self.nx,self.nx,6))

        for i in range(i_min,i_max):

            cutout_i = cutout_websky(i,name=self.name,name_mask=self.name_mask,
            mask_cib=self.mask_cib,sigma_mask=self.sigma_mask,inpaint_flag=self.inpaint_flag,mode=self.mode)

            dust = dust + cutout_i.dust
            synchro = synchro + cutout_i.synchro
            cib = cib + cutout_i.cib
            ksz = ksz + cutout_i.ksz
            noise = noise + cutout_i.noise
            cmb = cmb + cutout_i.cmb
            cib_masked = cib_masked + cutout_i.cib_masked
            #noise_no_ptf = noise_no_ptf + cutout_i.noise_no_ptf

        cutout_i.dust = dust/float(i_max-i_min)
        cutout_i.synchro = synchro/float(i_max-i_min)
        cutout_i.cib = cib/float(i_max-i_min)
        cutout_i.ksz = ksz/float(i_max-i_min)
        cutout_i.noise = noise/float(i_max-i_min)
        cutout_i.cmb = cmb/float(i_max-i_min)
        cutout_i.cib_masked = cib_masked/float(i_max-i_min)
    #    cutout_i.noise_no_ptf = noise_no_ptf/float(i_max-i_min)

        return cutout_i

#everything in muK

class cutout_websky:

    def __init__(self,i,name="maps/websky_maps/t_maps/",
    name_mask="maps/planck_maps/planck_field_",mask_cib=False,sigma_mask=None,inpaint_flag=False,mode="real",
    path = "/Users/user/Desktop/"):

        #name_mask = path + name_mask
        #name = path + name

        [mask_galaxy,mask_point,mask_tile] = np.load(name_mask + str(i) + "_mask.npy")

        self.mask_galaxy = mask_galaxy
        self.mask_point = mask_point
        self.mask_tile = mask_tile

        self.dust = np.load(name + "_dust_" + str(i) + "_tmap.npy")[0,:,:,:]
        self.synchro = np.load(name + "_synchro_" + str(i) + "_tmap.npy")[0,:,:,:]
        self.cib = np.load(name + "_cib_" + str(i) + "_tmap.npy")[0,:,:,:]

        if mode == "websky":

            self.tsz = np.load(name + "_tsz_" + str(i) + "_tmap.npy")[0,:,:,:]

        elif mode == "injected": #can be injected or injected_flat

            self.tsz = np.load(name + "_tsz_" + mode + "_" + str(i) + "_tmap.npy")[0,:,:,:]

        elif mode == "injected_flat":

            self.tsz = np.load(name + "_tsz_" + mode + "_" + str(i) + "_tmap.npy")

        self.ksz = np.load(name + "_ksz_" + str(i) + "_tmap.npy")[0,:,:,:]
        self.noise = np.load(name + "_noise_" + str(i) + "_tmap.npy")[0,:,:,:]
        self.cmb = np.load(name + "_cmb_" + str(i) + "_tmap.npy")[0,:,:,:]
        self.cib_masked = np.load(name + "_cib_masked_" + str(i) + "_tmap.npy")[0,:,:,:]
    #    self.noise_no_ptf = np.load(name + "_noise_no_ptf_" + str(i) + "_tmap.npy")[0,:,:,:]
        self.mask_point_cib = np.load(path + "maps/websky_maps/cib_mask_" + str(i) + ".npy")

        if mask_cib == True:

            self.cib = maps.mask_sigma_freq(self.cib,sigma_mask,inpaint_flag=inpaint_flag)

        self.nx = 1024
        self.l = 14.8  #in deg
        self.dx = self.l/self.nx/180.*np.pi
        self.pix = maps.pixel(self.nx,self.dx)

    def get_tmap(self,mode="all"):

        if mode == "all":

            ret = self.ksz + self.dust + self.synchro + self.cmb + self.noise + self.tsz # + self.cib_masked #

        elif mode == "extrag":

            ret = self.tsz + self.ksz + self.cmb + self.noise #+ self.cib_masked

        elif mode == "extrag_noise":

            ret = self.ksz + self.cmb + self.noise

        elif mode == "cmb+noise":

            ret = self.tsz + self.cmb + self.noise

        elif mode == "cmb+noise-tsz":

            ret = self.cmb + self.noise

        elif mode == "cmb+noise+ksz":

            ret = self.tsz + self.cmb + self.noise

        elif mode == "all-tsz":

            ret = self.ksz + self.dust + self.synchro + self.cmb + self.noise + self.cib_masked

        return ret



class detections_websky:

    def __init__(self,i_min=0,i_max=100,mode="all",indices=None):

        self.catalogue_obs = cat.cluster_catalogue()
        self.catalogue_obs_noit = cat.cluster_catalogue()
        self.catalogue_true = cat.cluster_catalogue()

        cutout = cutout_websky(0)
        pix = cutout.pix

        cutouts = cutouts_websky()

        apply_mask = True

        if indices is None:

            indices = range(i_min,i_max)
            self.sigma_matrix = np.zeros((i_max-i_min,5))

        else:

            self.sigma_matrix = np.zeros((len(indices),5))

        j = 0

        for i in indices:

            if mode == "all":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_all_" + str(i) + ".npy"

            if mode == "all+cib":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cib_" + str(i) + ".npy"

            if mode == "all+cib_mean":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cib_mean_" + str(i) + ".npy"

            if mode == "all+cib_random_2200":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cib_random_2200_" + str(i) + ".npy"

            if mode == "all+cib_random":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cib_random_" + str(i) + ".npy"

            elif mode == "cmb+noise":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cmb_" + str(i) + ".npy"

            if mode == "cmb+noise_tnoi":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cmb_tnoi_" + str(i) + ".npy"

            if mode == "all+cib_random_tnoi":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cib_random_tnoi_" + str(i) + ".npy"

            if mode == "cmb+noise_tnoi_nops":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cmb_tnoi_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "all+cib_random_tnoi_nops":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_all_cmb_random_ßtnoi_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "cmb+noise_tnoi_nops_l2":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cmb_tnoi_nops_l2_" + str(i) + ".npy"
                apply_mask = False

            if mode == "cmb+noise_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cmb_tnoi_nops_newf_" + str(i) + ".npy"
                apply_mask = False

            if mode == "all+cib_random_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_all_cib_random_tnoi_nops_newf_" + str(i) + ".npy"
                apply_mask = False

            if mode == "cmb+noise_tnoi_nops_newf_fixed":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_cmb_tnoi_nops_newf_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "noise_tnoi_nops_newf_fixed":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_wn_tnoi_nops_newf_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "all_cib_random_tnoi_nops_newf_fixed":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_all_cib_random_tnoi_nops_newf_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "wn_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_wn_tnoi_nops_newf_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_wn_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_wn_tnoi_nops_newf_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all+cib_random_tnoi_nops_newf": #0-100 and 667-768

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_all_cib_random_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_cmb+noise_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_cmb_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_wn_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_wn_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_all_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "all_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_all_tnoi_nops_newf_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all+cib_random_mean_tnoi_nops_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_all+cib_random_mean_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_wn_tnoi_nops_newf_nof":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_wn_tnoi_newf_nops_nof_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_wn_tnoi_nops_newf_fixed":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_wn_tnoi_newf_nops_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_wn_tnoi_nops_newf_noptf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_wn_tnoi_newf_nops_noptf_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_cmb+noise_tnoi_nops_newf_fixed":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_cmb_tnoi_newf_nops_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all+cib_random_tnoi_nops_newf_fixed":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_all+cib_random_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all+cib_random_tnoi_nops_newf_fixed_4ch":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_all+cib_random_tnoi_newf_nops_4ch_" + str(i) + ".npy"
                apply_mask = False

            if mode == "all_cib_random_tnoi_newf":

                name = "/Users/user/Desktop/catalogues/websky/catalogue_all+cib_random_tnoi_newf_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all_cib_random_cmb_comp":

                name = "/Users/user/Desktop/catalogues/websky/component_cmb_inj_all+cib_random_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all_cib_random_noi_comp":

                name = "/Users/user/Desktop/catalogues/websky/component_noi_inj_all+cib_random_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all_cib_random_cib_comp":

                name = "/Users/user/Desktop/catalogues/websky/component_cib_inj_all+cib_random_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all_cib_random_synchro_comp":

                name = "/Users/user/Desktop/catalogues/websky/component_synchro_inj_all+cib_random_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all_cib_random_dust_comp":

                name = "/Users/user/Desktop/catalogues/websky/component_dust_inj_all+cib_random_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all_cib_random_ksz_comp":

                name = "/Users/user/Desktop/catalogues/websky/component_ksz_inj_all+cib_random_tnoi_newf_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all+cib_random_nops_newf_fixed": #0-100 and 667-768

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_all+cib_random_nops_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_all+cib_random_nops_newf": #0-100 and 667-768

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_all+cib_random_nops_find_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_extr+cib_random_tnoi_nops_newf": #0-768

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr+cib_random_tnoi_nops_find_"  + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_extr+cib_random_tnoi_nops_newf_qttrue": #0-768

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr+cib_random_tnoi_nops_find_qttrue_"  + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_extr+cib_random_tnoi_nops_newf_qttrue_flat": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr+cib_random_tnoi_nops_find_qttrue_flat_"  + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_extr+cib_random_tnoi_nops_newf_flat": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr+cib_random_tnoi_nops_find_flat_"  + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_extr+cib_random_tnoi_nops_newf_tsubgrid_flat": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr+cib_random_tnoi_nops_find_tsubgrid_flat_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_extr+cib_random_tnoi_nops_newf_tsubgrid_findnosubgrid_flat": #0-100, 667-768

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr+cib_random_tnoi_nops_find_tsubgrid_findnosubgrid_flat_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_tnoi_nops_newf_fixed": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_tnoi_nops_newf_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_tnoi_nops_newf_fixed": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_random_tnoi_nops_newf_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_tnoi_nops_newf_find_tsubgrid_findnosubgrid": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_tnoi_nops_newf_find_tsubgrid_findnosubgrid_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_tnoi_nops_newf_find_tsubgrid_findnosubgrid_peak": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_tnoi_nops_newf_find_tsubgrid_findnosubgrid_peak_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_nops_newf_find_tsubgrid_findnosubgrid_peak": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_nops_newf_find_tsubgrid_findnosubgrid_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_nops_newf_tsubgrid_fixed": #0-100, 667-678

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_nops_newf_find_tsubgrid_fixed_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_nops_newf_tsubgrid_fixed_tszps_subtracted": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_nops_newf_find_tsubgrid_fixed_tszps_subtracted_" + str(i) + ".npy"

            if mode == "extrag_cib_random_nops_newf_tsubgrid_fixed_tszps_subtracted_noit": #0-200

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_nops_newf_find_tsubgrid_fixed_tszps_subtracted_noit_" + str(i) + ".npy"
                apply_mask = False


            if mode == "extrag_cib_random_nops_newf_tsubgrid_fixed_5": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_nops_newf_find_tsubgrid_fixed_5_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_nops_newf_tsubgrid_fixed_noit": #0-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_nops_newf_find_tsubgrid_fixed_noit_" + str(i) + ".npy"

                if i >= 100:

                    name = "/Users/user/Desktop/catalogues/websky/extrag_cib_random_nops_newf_tsubgrid_fixed_noit_" + str(i) + ".npy"

                apply_mask = False

            if mode == "extrag_cib_random_nops_newf_tsubgrid_fixed_tszps_subtracted_masked": #0-200 100-200 at q=5

                name = "/Users/user/Desktop/catalogues/websky/extrag_cib_random_nops_newf_tsubgrid_fixed_tszps_subtracted_masked_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_nops_newf_tsubgrid_fixed_tszps_subtracted_masked_b":

                name = "/Users/user/Desktop/catalogues/websky/extrag_cib_random_nops_newf_tsubgrid_fixed_tszps_subtracted_masked_b_" + str(i) + ".npy"
                apply_mask = False

            if mode == "extrag_cib_random_nops_newf_tsubgrid_fixed_5_subtracted_masked": #0-200

                name = "/Users/user/Desktop/catalogues/websky/catalogue_extrag_cib_nops_newf_find_tsubgrid_fixed_5_subtract_masked_" + str(i) + ".npy"
                apply_mask = False

            if mode == "true_tsz_random_tsubgrid_fixed": #100-200

                name = "/Users/user/Desktop/catalogues/websky/true_tsz_random_tsubgrid_fixed" + str(i) + ".npy"
                apply_mask = False

            if mode == "true_tsz_random_tsubgrid_fixed_masked": #100-200

                name = "/Users/user/Desktop/catalogues/websky/true_tsz_random_tsubgrid_fixed_masked_" + str(i) + ".npy"
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_fixed_subgrid": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_fixed_nosubgrid_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_fixed_nosubgrid": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_fixed_subgridnot_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_fixed_qtrue": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_fixed_qtrue_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_fixed_subgrid_qtrue": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_fixed_subgrid_qtrue_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_find_subgrid_qtrue": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_find_subgrid_qtrue_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_fixed_subgrid_noit": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_fixed_noit_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            #From here on: new apodisation

            if mode == "inj_extr_cib_random_nops_tnoi_find_apodnew": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_tnoi_find_apodnew_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_tnoi_fixed_apodnew": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_tnoi_fixed_apodnew_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_it_find_apodnew": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_it_find_apodnew_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_noit_find_apodnew": #00-100

                name = "/Users/user/Desktop/catalogues/websky/catalogue_inj_extr_cib_random_nops_noit_find_apodnew_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_tnoi_find_apodnew_dbscan": #00-100

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_tnoi_find_apodnew_dbscan_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_it_fixed_apodnew": #00-100

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_it_fixed_apodnew_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_noit_fixed_apodnew": #00-100

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_noit_fixed_apodnew_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_tnoi_find_apodnew": #00-100

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_tnoi_find_apodnew_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_tnoi_fixed_qtrue_apodnew": #00-100

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_tnoi_fixed_qtrue_apodnew_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            #End of new apodisation

            if mode == "inj_extr_cib_random_nops_tnoi_find_apodold_dbscan": #00-100

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_tnoi_find_apodold_dbscan_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_tnoi_find_apodold": #00-100

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_tnoi_find_apodold_dbscan_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_tnoi_find_apodold_15": #all

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_tnoi_find_apodold_15_" + str(i) + ".npy" #it's fixed
                apply_mask = False

                if i >= 100:

                    name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_tnoi_find_apodold_" + str(i) + ".npy" #it's fixed
                    apply_mask = False

            if mode == "inj_extr_cib_random_nops_noit_fixed_apodold": #all

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_noit_fixed_apodold_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_it_find_apodold_15": #all


                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_it_find_apodold_" + str(i) + ".npy" #it's fixed


                apply_mask = False

            if mode == "inj_extr_cib_random_nops_it_fixed_apodold": #all

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_it_fixed_apodold_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_tnoi_fixed_apodold": #all

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_tnoi_fixed_apodold_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_noit_find_apodold_15": #all

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_noit_find_apodold_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if mode == "inj_extr_cib_random_nops_it_fixed_apodold_truemasked": #all

                name = "/Users/user/Desktop/catalogues/websky/inj_extr_cib_random_nops_it_fixed_apodold_truemasked_" + str(i) + ".npy" #it's fixed
                apply_mask = False

            if os.path.isfile(name) == True:

                #print(i)

                results_new,a = np.load(name,allow_pickle=True)
                results_new.get_lonlat(i,pix)
                results_new.results_true.pixel_ids = np.ones(len(results_new.results_true.q_opt))*i #comment out when this is sorted out in code
                results_new.results_true.get_lonlat(8,i,pix) #idem
                self.catalogue_obs.append(results_new.results_refined)
                self.catalogue_obs_noit.append(results_new.results_refined_noit)

                results_true = results_new.results_true

                if apply_mask == True:

                    mask_select = np.load("/Users/user/Desktop/maps/websky_maps/mask_select/mask_select_" + str(i) + ".npy")
                    results_true = cat.apply_mask_select(results_true,mask_select,pix)

                self.catalogue_true.append(results_true)

                #print(i,len(results_new.results_true.q_opt),len(results_new.results_refined.q_opt))
                #if len(results_new.results_true.q_opt) < 20:

                #    print(i,len(results_new.results_true.q_opt))

                #self.sigma_matrix[j,:] = results_new.sigma_vec
                self.theta_500_sigma = results_new.theta_500_sigma

                j += 1

class detections_websky_paper:

    def __init__(self,data_label,noise_find_label,suffix="",i_min=0,i_max=100,
    indices=None,cib="random",name=None,path="/Users/user/Desktop/catalogues_def/websky/"):

        if cib == "random":

            self.prename = "paper_extr_cib_random_nops"

        elif cib == "":

            self.prename = "paper_extr_cib_nops"

        self.data_label = data_label
        self.noise_find_label = noise_find_label

        if name is None:

            full_name = path + self.prename + "_" + self.noise_find_label + suffix + "_"  + self.data_label

        else:

            full_name = path + name

        print(full_name)

        self.catalogue_obs = cat.cluster_catalogue()
        self.catalogue_obs_noit = cat.cluster_catalogue()
        self.catalogue_true = cat.cluster_catalogue()

        cutout = cutout_websky(0)
        pix = cutout.pix

        if indices is None:

            indices = range(i_min,i_max)
            self.sigma_matrix = np.zeros((i_max-i_min,15))

        else:

            self.sigma_matrix = np.zeros((len(indices),15))

        j = 0

        for i in indices:

            name = full_name + "_"  + str(i) + ".npy"

            if os.path.isfile(name) == True:

                #print(i)

                results_new,a = np.load(name,allow_pickle=True)
                cat.convert_results(results_new)
                results_new.get_lonlat(i,pix)
                results_new.results_true.catalogue["pixel_ids"] = np.ones(len(results_new.results_true.catalogue["q_opt"]))*i #comment out when this is sorted out in code
                results_new.results_true.get_lonlat(8,i,pix) #idem

                self.catalogue_obs.append(results_new.results_refined,append_keys="new")
                self.catalogue_obs_noit.append(results_new.results_refined_noit,append_keys="new")

                results_true = results_new.results_true

                self.catalogue_true.append(results_true,append_keys="new")

            #    if len(results_new.results_refined.catalogue["q_opt"]) != len(results_new.results_true.catalogue["q_opt"]):

                #    print(i,len(results_new.results_refined.catalogue["q_opt"]),len(results_new.results_true.catalogue["q_opt"]))
                #if len(results_new.results_true.q_opt) < 20:

                #    print(i,len(results_new.results_true.q_opt))

                self.sigma_matrix[j,:] = results_new.sigma_vec
                self.theta_500_sigma = results_new.theta_500_sigma

                j += 1

class detections_planck:

    def __init__(self,name,i_min=0,i_max=100,indices=None):

        self.name = name

        self.catalogue_obs = cat.cluster_catalogue()
        self.catalogue_obs_noit = cat.cluster_catalogue()

        cutout = cutout_websky(0)
        pix = cutout.pix

        if indices is None:

            indices = range(i_min,i_max)
            self.sigma_matrix = np.zeros((i_max-i_min,15))

        else:

            self.sigma_matrix = np.zeros((len(indices),15))

        j = 0

        full_name = "/Users/user/Desktop/catalogues_def/websky/" + name

        for i in indices:

            name = full_name + "_"  + str(i) + ".npy"

            if os.path.isfile(name) == True:

                #print(i)

                results_new,a = np.load(name,allow_pickle=True)
                results_new.get_lonlat(i,pix)
                self.catalogue_obs.append(results_new.results_refined)
                self.catalogue_obs_noit.append(results_new.results_refined_noit)

                self.sigma_matrix[j,:] = results_new.sigma_vec
                self.theta_500_sigma = results_new.theta_500_sigma

                j += 1

class detections_websky_processed:

    def __init__(self,data_label,noise_find_label,name_addition="",pixel_ids=None,cib="random",name=None):

            if cib == "random":

                prename = "paper_extr_cib_random_nops_" + noise_find_label + "_" + name_addition + data_label

            elif cib == "":

                prename = "paper_extr_cib_nops_" + noise_find_label + "_" + name_addition + data_label

            if name is not None:

                prename = name

            full_name = "/Users/user/Desktop/catalogues_def/processed_websky_paper/" + prename  + ".npy"
            (catalogue_obs,catalogue_true,metadata) = np.load(full_name,allow_pickle=True)

            print(full_name)

            catalogue_obs = cat.convert_catalogue(catalogue_obs)
            catalogue_true = cat.convert_catalogue(catalogue_true)

            self.catalogue_obs = catalogue_obs
            self.catalogue_true = catalogue_true

            if pixel_ids is not None:

                catalogue_obs_ret = cat.cluster_catalogue()
                catalogue_true_ret = cat.cluster_catalogue()

                for i in range(0,len(pixel_ids)):

                    indices = np.where(self.catalogue_obs.catalogue["pixel_ids"] == pixel_ids[i])
                    catalogue_obs_ret.append(cat.get_catalogue_indices(self.catalogue_obs,indices))
                    catalogue_true_ret.append(cat.get_catalogue_indices(self.catalogue_true,indices))

                self.catalogue_obs = catalogue_obs_ret
                self.catalogue_true = catalogue_true_ret

class detections_planck_processed:

    def __init__(self,name,pixel_ids=None):

        full_name = "/Users/user/Desktop/catalogues_def/processed_planck/" + name  + ".npy"
        (catalogue_obs,catalogue_obs_noit,metadata) = np.load(full_name,allow_pickle=True)

        self.catalogue_obs = catalogue_obs
        self.catalogue_obs_noit = catalogue_obs_noit

        if pixel_ids is not None:

            catalogue_obs_ret = cat.cluster_catalogue()
            catalogue_obs_noit_ret = cat.cluster_catalogue()

            for i in range(0,len(pixel_ids)):

                indices = np.where(self.catalogue_obs.catalogue["pixel_ids"] == pixel_ids[i])
                catalogue_obs_ret.append(cat.get_catalogue_indices(self.catalogue_obs,indices))
                catalogue_obs_noit_ret.append(cat.get_catalogue_indices(self.catalogue_obs_noit,indices))

            self.catalogue_obs = catalogue_obs_ret
            self.catalogue_obs_noit = catalogue_obs_noit_ret


def draw_circles(catalogue_tile,img,pix,scaling=1.,theta_500_units="rad",
save_name=None,plot=True,cmap=pl.get_cmap("RdYlBu"),pixel_id=0,title=None):

    nx = pix.nx
    dx = pix.dx
    extent_arcmin = nx*dx*180.*60./np.pi

    theta_x = catalogue_tile.catalogue["theta_x"]
    theta_y = catalogue_tile.catalogue["theta_y"]
    theta_500 = catalogue_tile.catalogue["theta_500"]#c
    #theta_500 = catalogue_tile.theta_200m

    if theta_500_units == "arcmin":

        theta_500 = theta_500/180./60.*np.pi

    theta_500 = theta_500*180.*60./np.pi

    pl.rc('text', usetex=True)
    pl.rc('font', family='serif')
    fig,ax = pl.subplots(1)
    ax.set_aspect('equal')
    ax.set_xlabel("arcmin")
    ax.set_ylabel("arcmin")

    if title is not None:

        ax.set_title(title)

    img_file = ax.imshow(img,cmap=cmap,extent=[0,extent_arcmin,0,extent_arcmin],interpolation=None)
    #fig.colorbar(img_file,ax=ax)

    xc = theta_x*180.*60./np.pi
    yc = ((theta_y))*180.*60./np.pi
    r = theta_500/scaling#/dx/scaling

    for i in range(0,len(theta_500)):

        circ = Circle((xc[i],yc[i]),r[i],fill=False)
        ax.add_patch(circ)

    if save_name is not None:

        pl.savefig(save_name)

    if plot == True:

        pl.show()

    pl.close()


def draw_circles_subplot(catalogue_tile,ax,pix,scaling=1.,theta_500_units="rad"):

    nx = pix.nx
    dx = pix.dx

    theta_x = catalogue_tile.catalogue["theta_x"]
    theta_y = catalogue_tile.catalogue["theta_y"]
    theta_500 = catalogue_tile.catalogue["theta_500"]#c
    #theta_500 = catalogue_tile.theta_200m

    if theta_500_units == "arcmin":

        theta_500 = theta_500/180./60.*np.pi

    ax.set_aspect('equal')

    xc = theta_x/dx
    yc = nx - (theta_y/dx)
    r = theta_500/dx/scaling

    for i in range(0,len(theta_500)):

        circ = Circle((xc[i],yc[i]),r[i],fill=False)
        ax.add_patch(circ)

def draw_circles_lonlat(catalogue_tile,img,pix,scaling=1.,theta_500_units="rad",
save_name=None,plot=True,cmap=pl.get_cmap("RdYlBu"),pixel_id=0):

    nx = pix.nx
    dx = pix.dx

    theta_x = catalogue_tile.catalogue["theta_x"]
    theta_y = catalogue_tile.catalogue["theta_y"]
    theta_500 = catalogue_tile.catalogue["theta_500"]#c
    #theta_500 = catalogue_tile.theta_200m


    if theta_500_units == "arcmin":

        theta_500 = theta_500/180./60.*np.pi

    fig,ax = pl.subplots(1)
    ax.set_aspect('equal')

    (lon_bl,lat_bl,lon_br,lat_br,lon_tl,lat_tl,lon_tr,lat_tr) = sphere.get_field_limits(8,pixel_id,nx,dx)

    ax.imshow(img,cmap=cmap,extent=[lon_bl,lon_br,lat_bl,lat_tl],interpolation='none')#,vmin=-3, vmax=8)
    ax.set_aspect(abs((lon_bl-lon_br)/(lat_bl-lat_tl)))
    xc = theta_x/dx
    yc = nx - (theta_y/dx)
    r = theta_500/dx/scaling

    xc = catalogue_tile.catalogue["lon"]
    yc = catalogue_tile.catalogue["lat"]
    r = theta_500/np.pi*180.

    print(xc)

    print(r)


    for i in range(0,len(theta_500)):

        circ = Circle((xc[i],yc[i]),r[i],fill=False)
        ax.add_patch(circ)

    if save_name is not None:

        pl.savefig(save_name)

    if plot == True:

        pl.show()

    pl.close()
