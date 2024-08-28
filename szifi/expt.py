import numpy as np
import healpy as hp
from astropy.io import fits
from szifi import params, sed

class experiment:

    def __init__(self,experiment_name,params_szifi=params.params_szifi_default):

        self.experiment_name = experiment_name
        self.params_szifi = params_szifi
        self.transmission_list = None

        if self.experiment_name == "Planck_real":

            self.nu_eff = np.array([101.31,142.710,221.915,361.290,557.53,862.68])*1e9
            self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.*1.7508e4,0.72/1000.*6.9653e5])*60. #all in muK arcmin

            self.FWHM = np.array([9.68,7.3,5.02,4.94,4.83,4.64]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
            beams_file = fits.open(params_szifi["path"] + "data/HFI_RIMO_Beams-075pc_R2.00.fits")
            indices_beams = [3,4,5,6,11,12]

            self.beams = []

            for i in range(0,len(self.nu_eff)):

                self.beams.append(beams_file[indices_beams[i]].data[0][0])

            self.tsz_sed_paper = np.array([1./(-0.24815),1./(-0.35923),1./5.152,1./0.161098,1./0.06918,1./0.038])*1e6 #tSZ signature in muK

            tsz_sed_model = sed.tsz_model()
            self.tsz_sed = tsz_sed_model.get_sed_exp_bandpass(self)
            self.bandpass_file = fits.open(params_szifi["path"] + "data/HFI_RIMO_Beams-075pc_R2.00.fits")

        if self.experiment_name == "Planck_validation":

            self.nu_eff = np.array([100.,143.,217.,353.,545.,857.])*1e9
            self.FWHM = np.array([9.68,7.3,5.02,4.94,4.83,4.64]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
            self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.*1.7508e4,0.72/1000.*6.9653e5])*60.
            self.MJysr2muK = np.array([4.1877e3,2.632e3,2.0676e3,3.371e3,1.7508e4,6.9653e5])

            self.tsz_sed_old = np.array([-4.1103e6,-2.8355e6,-2.1188e4,6.1071e6,1.5257e7,3.0228e7])
            self.tsz_sed = sed.tsz_model().get_sed(self.nu_eff)

        if self.experiment_name == "Planck_simple":

            self.nu_eff = np.array([100.,143.,217.,353.,545.,857.])*1e9
            self.FWHM = np.array([9.68,7.3,5.02,4.94,4.83,4.64]) #FWHM1 of https://www.aanda.org/articles/aa/pdf/2016/10/aa25820-15.pdf
            self.noise_levels = np.array([1.29,0.55,0.78,2.56,0.78/1000.*1.7508e4,0.72/1000.*6.9653e5])*60.
            self.MJysr2muK = np.array([4.1877e3,2.632e3,2.0676e3,3.371e3,1.7508e4,6.9653e5])
            self.tsz_sed = np.array([-4.1103e6,-2.8355e6,-2.1188e4,6.1071e6,1.5257e7,3.0228e7])

        elif self.experiment_name == "SPTpol_simple":

            self.nu_eff = np.array([93.,145])*1e9
            self.FWHM = np.array([1.7,1.2])

            self.area_patches = np.array([276.,250.,275.2,251.8,272.9,248.8,277,250.3,274.3,270.8]) #from https://arxiv.org/pdf/1910.04121.pdf
            self.sigma95 = np.array([61.3,59.4,80.4,61.5,54.6,43.8,57.0,54.8,77.6,50.7])
            self.sigma150 = np.array([30.5,36.6,39.2,36.6,28.6,25.3,31.4,31.6,40.,30.])
            self.noise_levels = np.array([np.average(self.sigma95,weights=self.area_patches),np.average(self.sigma150,weights=self.area_patches)])

        elif self.experiment_name == "ACT_simple": #as used in Hilton et al. 2020. Why not 220 GHz?

            self.nu_eff = np.array([100.,145.])*1e9
            self.FWHM = np.array([2.2,1.4])
            self.noise_levels = np.array([33.,27.]) #Guess, based on Fig. 14 of https://arxiv.org/pdf/2007.07290.pdf

        elif self.experiment_name == "SObaseline_simple": #from https://arxiv.org/pdf/1808.07445.pdf

            self.nu_eff = np.array([27.,39.,93.,145.,225.,278.])*1e9
            self.FWHM = np.array([7.4,5.1,2.2,1.4,1.,0.9])
            self.noise_levels = np.array([71.,36.,8.,10.,22.,54.])
            self.MJysr2muK = np.array([4.1877e3,2.632e3,2.0676e3,3.371e3,1.7508e4,6.9653e5])
            self.tsz_sed = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-2.7685e6,3.1517e5,2.7314e6])

        elif self.experiment_name == "SOgoal_simple": #from https://arxiv.org/pdf/1808.07445.pdf

            self.nu_eff = np.array([27.,39.,93.,145.,225.,278.])*1e9
            self.FWHM = np.array([7.4,5.1,2.2,1.4,1.,0.9])
            self.noise_levels = np.array([52.,27.,5.8,6.3,15.,37.])
            self.tsz_sed = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-2.7685e6,3.1517e5,2.7314e6])

        self.n_freqs = len(self.nu_eff)

    #Only if beam in self.beam is wanted

    def get_beam(self,i,ptf=True,lmax=4000,nside=2048):

        ell = np.arange(0,lmax+1)
        beam = self.beams[i]

        if ptf == True:

            beam = beam*hp.sphtfunc.pixwin(nside)[0:4001]

        return ell,beam

    def get_ptf(self,nside):

        ptf = hp.sphtfunc.pixwin(nside)
        ell = np.arange(0,len(ptf))

        return ell,ptf

    def get_band_transmission(self):

        if self.experiment_name == "Planck_real":

            bandpass_file = fits.open(self.params_szifi["path"] + "data/HFI_RIMO_R3.00.fits")
            channel_indices = [3,4,5,6,7,8]

            self.transmission_list = []
            self.nu_transmission_list = [] #in Hz

            for i in range(0,len(channel_indices)):

                data_vector = bandpass_file[channel_indices[i]].data

                len_data = len(data_vector)
                wavelength = np.zeros(len_data)
                transmission = np.zeros(len_data)

                for j in range(0,len_data):

                    wavelength[j] = data_vector[j][0]
                    transmission[j] = data_vector[j][1]

                c_light = 299792458.
                nu = wavelength*1e2*c_light

                self.transmission_list.append(transmission[1:])
                self.nu_transmission_list.append(nu[1:])

class websky_conversions:

    def __init__(self):

        self.freqs = np.array([27.,39.,93.,100.,143.,145.,217.,225.,280.,353.,545.,857.])*1e9
        self.MJysr2muK = np.array([4.5495e4,2.2253e4,4.6831e3,4.1877e3,2.6320e3,2.5947e3,2.0676e3,2.0716e3,2.3302e3,3.3710e3,1.7508e4,6.9653e5])
        self.y2muK = np.array([-5.3487e6,-5.2384e6,-4.2840e6,-4.1103e6,-2.8355e6,-2.7685e6,-2.1188e4,3.1517e5,2.7314e6,6.1071e6,1.5257e7,3.0228e7])

    def get_MJysr2muK(self,freq):

        return np.interp(freq,self.freqs,self.MJysr2muK)

    def get_y2muK(self,freq):

        return np.interp(freq,self.freqs,self.y2muK)
