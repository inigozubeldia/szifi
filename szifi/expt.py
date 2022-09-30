import numpy as np
import maps
from astropy.io import fits
import pylab as pl
import healpy as hp
import model

class planck_specs():

    def __init__(self,path="/Users/user/Desktop/"):

        self.tsz_f_nu_K = np.array([1./(-0.24815),1./(-0.35923),1./5.152,1./0.161098,1./0.06918,1./0.038]) #tSZ signature in units of K
        self.K2mK = np.array([1e6,1e6,1e6,1e6,1.,1.])
        self.nu_eff = np.array([101.31,142.710,221.915,361.290,557.53,862.68])*1e9
        self.unit_conversion = np.array([1.,1.,1.,1.,58.04,2.27])  #from K to MJy/sr
        self.tsz_f_nu = self.tsz_f_nu_K*1e6 #tSZ signature in units of muK
    #    self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.,0.72/1000.])*60. #first 4 in muK arcmin, last 2 in MJk/sr arcmin.
        self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.*1.7508e4,0.72/1000.*6.9653e5])*60. #all in muK arcmin
        self.n_freqs = 6
        self.FWHM = np.array([9.68,7.3,5.02,4.94,4.83,4.64]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
        #self.FWHM = np.array([0.,0.,0.,0.,0.,0.]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
        #self.FWHM = np.array([9.68,9.68,9.68,9.68,9.68,9.68]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
        self.beams = fits.open(path + "data/planck_data/HFI_RIMO_Beams-075pc_R2.00.fits")
        self.indices_beams = [3,4,5,6,11,12]
        self.name = "Planck"
        self.MJysr_to_muK_websky = np.array([4.1877e3,2.6320e3,2.0676e3,3.3710e3,1.7508e4,6.9653e5]) #from Websky paper
    #    self.tsz_f_nu = np.array([-4.1103e6,-2.8355e6,-2.1188e4,6.1071e6,1.5257e7,3.0228e7]) #from Websky paper

    def get_beam(self,i,ptf=True):

        ell = np.arange(0,4001)
        beam = self.beams[self.indices_beams[i]].data[0][0]

        if ptf == True:

            beam = beam*hp.sphtfunc.pixwin(2048)[0:4001]

        return ell,beam

    def get_ptf(self):

        ptf = hp.sphtfunc.pixwin(2048)
        ell = np.arange(0,len(ptf))

        return ell,ptf

    def get_class_sz_ps(self):

        [cl_sz,cl_cib_cib,cl_tsz_cib] = np.load("class_sz_tsz_cib_planck.py")

        return [cl_sz,cl_cib_cib,cl_tsz_cib]

class custom_specs():

    def __init__(self,path="/Users/user/Desktop/"):

        self.nu_eff = np.array([101.31,142.710,221.915,361.290,500,557.53,650,862.68,1000])*1e9
    #    self.nu_eff = np.array([101.31,142.710,221.915])*1e9
        self.noise_levels = np.ones(len(self.nu_eff))*3*60. #all in muK arcmin
        self.FWHM = np.ones(len(self.nu_eff))*5. #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf

        self.tsz_f_nu = model.get_tsz_f_nu(self.nu_eff,"muK") #tSZ signature in units of muK
        self.n_freqs = len(self.nu_eff)
        self.name = "custom"

    def get_ptf(self):

        ptf = hp.sphtfunc.pixwin(2048)
        ell = np.arange(0,len(ptf))

        return ell,ptf

class websky_specs():

    def __init__(self,path="/Users/user/Desktop/"):

        self.nu_eff_GHz = np.array([100,143,217,353,545,857])
        self.nu_eff = self.nu_eff_GHz*1e9
        self.MJysr2muK = np.array([4.1877e3,2.632e3,2.0676e3,3.371e3,1.7508e4,6.9653e5])
        self.y2muK = np.array([-4.1103e6,-2.8355e6,-2.1188e4,6.1071e6,1.5257e7,3.0228e7])
        self.tsz_f_nu = self.y2muK
        self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.*1.7508e4,0.72/1000.*6.9653e5])*60. #all in muK arcmin
        self.n_freqs = 6
        self.beams = fits.open(path + "data/planck_data/HFI_RIMO_Beams-075pc_R2.00.fits")
        self.indices_beams = [3,4,5,6,11,12]
        self.MJysr_to_muK_websky = np.array([4.1877e3,2.6320e3,2.0676e3,3.3710e3,1.7508e4,6.9653e5]) #from Websky paper
        self.FWHM = np.array([9.68,7.3,5.02,4.94,4.83,4.64])
        
    def get_beam(self,i,ptf=True):

        i = i

        ell = np.arange(0,4001)
        beam = self.beams[self.indices_beams[i]].data[0][0]

        if ptf == True:

            beam = beam*hp.sphtfunc.pixwin(2048)[0:4001]

        return ell,beam

    def get_ptf(self):

        ptf = hp.sphtfunc.pixwin(2048)
        ell = np.arange(0,len(ptf))

        return ell,ptf
