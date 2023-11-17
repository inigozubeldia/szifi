import numpy as np
import pylab as pl
import szifi
from astropy.io import fits
import scipy.integrate as integrate

#Check that SZiFi computes the tSZ SED correctly for Planck

tsz_sed_model = szifi.tsz_model()
exp = szifi.experiment("Planck_real")

params_szifi = szifi.params_szifi_default

exp.get_band_transmission()
transmision_list = exp.transmission_list
nu_transmission_list = exp.nu_transmission_list

nu = exp.nu_eff
tsz_sed_published = exp.tsz_sed
tsz_sed_bandpass = tsz_sed_model.get_sed_exp_bandpass(exp)
tsz_sed_published2 =np.array([-4.03,-2.78,0.19,6.21,14.46])*1e6

nu = nu/1e9

pl.scatter(nu,tsz_sed_published,label="Published")
pl.scatter(nu,tsz_sed_bandpass,label="tSZ SED with bandpass")
pl.xlabel("Frequency [GHz]")
pl.ylabel("SED")
pl.legend()
pl.savefig("tsz_sed_planck.pdf")
pl.show()

pl.scatter(nu,(tsz_sed_bandpass-tsz_sed_published)/tsz_sed_published*100.,label="Fractional difference")
pl.xlabel("Frequency [GHz]")
pl.ylabel("Fractional difference [%]")
pl.legend()
pl.savefig("tsz_sed_planck_error.pdf")
pl.show()
