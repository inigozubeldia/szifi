import numpy as np
from scipy import integrate
from szifi import params
from szifi.model import constants

class cib_model: #Modified blackbody, model used in Websky.

    def __init__(self,params_model=params.params_model_default):

        self.alpha = params_model["alpha_cib"]
        self.beta = params_model["beta_cib"]
        self.gamma = params_model["gamma_cib"]
        self.T0 = params_model["T0_cib"]
        self.z = params_model["z_eff_cib"]

        self.nu0 = 10000.
        self.nu_pivot = 1#3e9

        self.moments = {}

        self.const = constants()

    #Input nu is experiment nu (i.e. at z = 0).

    def get_sed_SI(self,nu=None):

        nup = nu*(1.+self.z)
        sed = (nup/self.nu_pivot)**self.beta*planckian(nup,self.T0*(1+self.z)**self.alpha)/1e13
        #normalisation so that amplitude is comparable to tSZ SED

        return sed

    #Input nu is experiment nu (i.e. at z = 0).

    def get_sed_muK(self,nu=None):

        sed = self.get_sed_SI(nu=nu)

        return sed/dBnudT(nu)

    def get_sed_muK_experiment(self,experiment=None,bandpass=False):

        if bandpass == False:

            sed = self.get_sed_muK(nu=experiment.nu_eff)

        elif bandpass == True:

            if experiment.transmission_list is None:

                experiment.get_band_transmission()

            sed = integrate_sed_bandpass(sed_func=self.get_sed_muK,exp=experiment)

        return sed

    def get_sed_first_moments_experiment(self,moment_parameters=None,bandpass=None,experiment=None):

        if bandpass == False:

            nu = experiment.nu_eff

            if "beta" in moment_parameters:

                self.moments["beta"] = self.get_sed_derivative_beta_muK(nu)

            if "betaT" in moment_parameters:

                self.moments["betaT"] = self.get_sed_derivative_betaT_muK(nu)

        elif bandpass == True:

            if experiment.transmission_list is None:

                experiment.get_band_transmission()

            if "beta" in moment_parameters:

                self.moments["beta"] = integrate_sed_bandpass(sed_func=self.get_sed_derivative_beta_muK,exp=experiment)

            if "betaT" in moment_parameters:

                self.moments["betaT"] = integrate_sed_bandpass(sed_func=self.get_sed_derivative_betaT_muK,exp=experiment)


    def get_sed_derivative_beta_muK(self,nu):

        nup = nu*(1.+self.z)
        der = self.get_sed_SI(nu=nu)*np.log(nup/self.nu_pivot)*MJysr_to_muK(nu)

        return der

    def get_sed_derivative_betaT_muK(self,nu):

        nup = nu*(1.+self.z)
        exponential = np.exp(self.const.h*nup/(self.const.k_B*self.T0*(1+self.z)**self.alpha))
        der = -2.*self.const.h*nup**(3.+self.beta)/(self.const.c_light**2*(exponential-1.)**2)*exponential*self.const.h*nup/self.const.k_B*MJysr_to_muK(nu)

        return der

#nu in Hz, T in K

def planckian(nu,T):

    const = constants()
    planck = 2.*const.h*nu**3/(const.c_light**2*(np.exp(const.h*nu/(const.k_B*T))-1.))

    return planck

#nu in Hz, T_CMB in K. To convert from Jy/sr to T_CMB, dvide by output of this function

def dBnudT(nu,T_CMB=constants().T_CMB):

    const = constants()

    x = const.h*nu/(const.k_B*T_CMB)
    f = x**2*np.exp(x)/(np.exp(x)-1.)**2.

    return 2.*const.k_B*nu**2/const.c_light**2*f

def MJysr_to_muK(nu):

    return 1./dBnudT(nu)*1e6/1e20

def muK_to_MJysr(nu):

    return dBnudT(nu)/1e6*1e20


#T_e is electron temperature, if T = None, non-relativistic tSZ SED is calculated
#nu in Hz

class tsz_model:

    def __init__(self,T_e=None):

        self.T_e = T_e
        self.const = constants()

    #Returns SED in muK

    def get_sed(self,nu):

        if self.T_e is None:

            x = nu*self.const.h/(self.const.k_B*self.const.T_CMB)
            SED = (x/np.tanh(0.5*x)-4.)*self.const.T_CMB*1e6

        return SED

    #Returns SED for a given experiment, integrated against the bandpasses

    def get_sed_exp_bandpass(self,exp):

        if exp.transmission_list is None:

            exp.get_band_transmission()

        sed_bandpass = integrate_sed_bandpass(sed_func=self.get_sed,exp=exp)

        return sed_bandpass

def integrate_sed_bandpass(sed_func=None,exp=None):

    n_freqs = len(exp.transmission_list)
    sed_bandpass = np.zeros(n_freqs)
    conversion = np.zeros(n_freqs)

    for i in range(0,n_freqs):

        transmission = exp.transmission_list[i]
        nu = exp.nu_transmission_list[i]
        sed = sed_func(nu)*muK_to_MJysr(nu)

        sed_bandpass[i] = integrate.simps(transmission*sed,nu)
        conversion[i] = integrate.simps(transmission*muK_to_MJysr(nu),nu)

    sed_bandpass = sed_bandpass/conversion

    return sed_bandpass
