import numpy as np
import maps
from astropy.io import fits
import pylab as pl
import healpy as hp
import model

class planck_specs():

    def __init__(self,path="/rds-d4/user/iz221/hpc-work/"):

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
        self.name = "Planck_real"
        self.MJysr_to_muK_websky = np.array([4.1877e3,2.6320e3,2.0676e3,3.3710e3,1.7508e4,6.9653e5]) #from Websky paper
    #    self.tsz_f_nu = np.array([-4.1103e6,-2.8355e6,-2.1188e4,6.1071e6,1.5257e7,3.0228e7]) #from Websky paper

    def get_beam(self,i,ptf=True,nside=2048,lmax=4000):

        ell = np.arange(0,lmax+1)
        beam = self.beams[self.indices_beams[i]].data[0][0]

        if ptf == True:

            beam = beam*hp.sphtfunc.pixwin(nside)[0:4001]

        return ell,beam

    def get_ptf(self,nside=2048):

        ptf = hp.sphtfunc.pixwin(nside)
        ell = np.arange(0,len(ptf))

        return ell,ptf

    def get_class_sz_ps(self):

        [cl_sz,cl_cib_cib,cl_tsz_cib] = np.load("class_sz_tsz_cib_planck.py")

        return [cl_sz,cl_cib_cib,cl_tsz_cib]

class custom_specs():

    def __init__(self,path="/rds-d4/user/iz221/hpc-work/"):

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

    def __init__(self,path="/rds-d4/user/iz221/hpc-work/"):

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
        self.name = "Planck_real"

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

class experiment_simple:

    def __init__(self,exp="planck",path="/rds-d4/user/iz221/hpc-work/"):

        self.experiment = exp_specs_simple(exp=exp)
        self.nu_eff = self.experiment.nu_eff
        self.FWHM = self.experiment.FWHM
        self.noise_levels = self.experiment.noise_levels
        self.name = self.experiment.name

        self.n_freqs = len(self.nu_eff)
        self.tsz_f_nu = model.get_tsz_f_nu(self.nu_eff,units="muK")


#nu_eff in GHz
#FWHM in arcmin
#noise levels in muK arcmin
#tsz_f_nu_k in muK

class exp_specs_simple:

    def __init__(self,exp="Planck"):

        if exp == "Planck":

            self.nu_eff = np.array([100.,143.,217.,353.,545.,857.])*1e9
            self.FWHM = np.array([9.68,7.3,5.02,4.94,4.83,4.64]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
            self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.*1.7508e4,0.72/1000.*6.9653e5])*60.
            self.MJysr2muK = np.array([4.1877e3,2.632e3,2.0676e3,3.371e3,1.7508e4,6.9653e5])
            self.y2muK = np.array([-4.1103e6,-2.8355e6,-2.1188e4,6.1071e6,1.5257e7,3.0228e7])

        elif exp == "SPTpol":

            #self.nu_eff = np.array([95.,150])*1e9 real values
            self.nu_eff = np.array([93.,145])*1e9
            self.FWHM = np.array([1.7,1.2])

            self.area_patches = np.array([276.,250.,275.2,251.8,272.9,248.8,277,250.3,274.3,270.8]) #from https://arxiv.org/pdf/1910.04121.pdf
            self.sigma95 = np.array([61.3,59.4,80.4,61.5,54.6,43.8,57.0,54.8,77.6,50.7])
            self.sigma150 = np.array([30.5,36.6,39.2,36.6,28.6,25.3,31.4,31.6,40.,30.])
            self.noise_levels = np.array([np.average(self.sigma95,weights=self.area_patches),np.average(self.sigma150,weights=self.area_patches)])

        elif exp == "ACT": #as used in Hilton et al. 2020. Why not 220 GHz?

            #self.nu_eff = np.array([98.,150.])*1e9 real values
            self.nu_eff = np.array([100.,145.])*1e9
            self.FWHM = np.array([2.2,1.4])
            self.noise_levels = np.array([33.,27.]) #Guess, based on Fig. 14 of https://arxiv.org/pdf/2007.07290.pdf


        elif exp == "SObaseline": #from https://arxiv.org/pdf/1808.07445.pdf

            self.nu_eff = np.array([27.,39.,93.,145.,225.,278.])*1e9
            self.FWHM = np.array([7.4,5.1,2.2,1.4,1.,0.9])
            self.noise_levels = np.array([71.,36.,8.,10.,22.,54.])
            self.MJysr2muK = np.array([4.1877e3,2.632e3,2.0676e3,3.371e3,1.7508e4,6.9653e5])
            self.y2muK = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-2.7685e6,3.1517e5,2.7314e6])

        elif exp == "SOgoal": #from https://arxiv.org/pdf/1808.07445.pdf

            self.nu_eff = np.array([27.,39.,93.,145.,225.,278.])*1e9
            self.FWHM = np.array([7.4,5.1,2.2,1.4,1.,0.9])
            self.noise_levels = np.array([52.,27.,5.8,6.3,15.,37.])
            self.y2muK = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-2.7685e6,3.1517e5,2.7314e6])

#        self.tsz_f_nu = self.y2muK
        self.n_freqs = len(self.nu_eff)
        self.name = exp

class websky_conversions:

    def __init__(self):

        self.freqs = np.array([27.,39.,93.,100.,143.,145.,217.,225.,280.,353.,545.,857.])*1e9
        self.MJysr2muK = np.array([4.5495e4,2.2253e4,4.6831e3,4.1877e3,2.6320e3,2.5947e3,2.0676e3,2.0716e3,2.3302e3,3.3710e3,1.7508e4,6.9653e5])
        self.y2muK = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-4.1103e6,-2.8355e6,-2.7685e6,-2.1188e4,3.1517e5,2.7314e6,6.1071e6,1.5257e7,3.0228e7])

    def get_MJysr2muK(self,freq):

        return np.interp(freq,self.freqs,self.MJysr2muK)

    def get_y2muK(self,freq):

        return np.interp(freq,self.freqs,self.y2muK)
