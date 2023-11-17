import numpy as np
import pylab as pl
import szifi
from astropy.io import fits

#Check that SZiFi computes the tSZ SED correctly for Planck

tsz_sed_model = szifi.tsz_model()
exp = szifi.experiment("Planck_real")

params_szifi = szifi.params_szifi_default

cib_model = szifi.cib_model()

sed_nueff = cib_model.get_sed_muK_experiment(experiment=exp,bandpass=False)
nu = exp.nu_eff
sed_bandpass = cib_model.get_sed_muK_experiment(experiment=exp,bandpass=True)

nu = nu/1e9

pl.scatter(nu,sed_nueff,label="No bandpass")
pl.scatter(nu,sed_bandpass,label="With bandpass")
pl.xlabel("Frequency [GHz]")
pl.ylabel("SED")
pl.legend()
pl.savefig("tsz_sed_cib.pdf")
pl.show()

pl.scatter(nu,(sed_nueff-sed_bandpass)/sed_bandpass*100.,label="Fractional difference")
pl.xlabel("Frequency [GHz]")
pl.ylabel("Fractional difference [%]")
pl.legend()
pl.savefig("tsz_sed_cib_error.pdf")
pl.show()
