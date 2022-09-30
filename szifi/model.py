import numpy as np
from astropy.cosmology import Planck15
from scipy import integrate
import maps
import expt
import scipy.optimize as optimize
import pylab as pl
import websky
import scipy
#from pysz import pysz

class gnfw_arnaud:

    def __init__(self,M_500,z_halo,cosmology,c_500=1.177,path="/Users/user/Desktop/"):

        #Create variables

        self.M_500 = M_500*1e15
        self.c = c_500
        self.z_halo = z_halo
        self.cosmology = cosmology
        self.path = path

        #Calculate basic quantities

        const = constants()
        self.rho_c = cosmology.critical_density(self.z_halo).value*1000.*const.mpc**3/const.solar
        self.r_s = (3./4.*self.M_500/self.rho_c/500./np.pi/self.c**3)**(1./3.)
        self.R_500 = self.c*self.r_s

        self.chi_h = cosmology.comoving_distance(self.z_halo).value #In Mpc
        self.d_ad_h = self.chi_h/(1.+z_halo)

        self.theta_500 = self.R_500/self.d_ad_h
        self.theta_500_arcmin = self.theta_500/np.pi*180.*60.
        self.my_params_sz = params_sz(cosmology,c=self.c)

        self.Ez = self.cosmology.H(self.z_halo).value/self.cosmology.H0.value
        self.h70 = self.cosmology.H0.value/70.
        self.P_500 = 1.65e-3*self.Ez**(8./3.)*(self.M_500/3e14*self.h70)**(2./3.)*self.h70**2  #in units of keV cm-3
        const = constants()
        self.P2Y = const.mpc*100.*const.sigma_thomson_e/const.mass_e #converts pressure in units of keV cm-3 to dimensionless Compton y
        self.Y_500 = self.P_500*self.R_500**3*4.*np.pi/3.*self.P2Y #characteristic Y, in units of Mpc^2


    def get_p_cal_map(self,pix,c_500,theta_misc=[0.,0.]):

        nx = pix.nx
        dx = pix.dx
        ny = pix.ny
        dy = pix.dy

        theta_max = np.sqrt((nx*dx)**2+(ny*dy)**2)*0.5*1.1
        theta_vec = np.linspace(0.0001,theta_max,1000)
        p_cal_vec = np.zeros(len(theta_vec))

        for i in range(0,len(theta_vec)):

            x = theta_vec[i]/self.theta_500
            p_cal_vec[i] = self.get_p_cal_int(self.my_params_sz,x)

        theta_map = maps.rmap(pix).get_distance_map_wrt_centre(theta_misc)
        p_cal_map = np.interp(theta_map,theta_vec,p_cal_vec)

        return p_cal_map

    def get_y_map(self,pix,theta_misc=[0.,0.]):

        p_cal_map = self.get_p_cal_map(pix,self.c,theta_misc)
        y_map = self.p_cal_to_y(p_cal_map)

        return y_map

    def get_y_norm(self,type="R_500"):  #returns y at R_500

        if type == "R_500":

            r = 1.

        elif type == "centre":

            r = 1e-6

        return self.p_cal_to_y(self.get_p_cal_int(self.my_params_sz,r))

    def get_y_at_angle(self,theta): #angle in rad

        return self.p_cal_to_y(self.get_p_cal_int(self.my_params_sz,theta/self.theta_500))

    def get_y_map_convolved(self,pix,fwhm_arcmin,theta_misc=[0.,0.]):

        return maps.get_gaussian_convolution(self.get_y_map(pix,theta_misc=theta_misc),fwhm_arcmin,pix)

    def get_y_norm_convolved(self,pix,fwhm_arcmin,y_map=None,theta_misc=[0.,0.]):

        if y_map == None:

            y_map = self.get_y_map_convolved(pix,fwhm_arcmin,theta_misc)

        (theta_x,theta_y) = theta_misc
        x_coord = maps.rmap(pix).get_x_coord_map_wrt_centre(theta_x)[0,:]/self.theta_500
        y_coord = maps.rmap(pix).get_y_coord_map_wrt_centre(theta_y)[:,0]/self.theta_500

        return interpolate.interp2d(x_coord,y_coord,y_map)(1.,0.)


    def p_cal_to_y(self,p_cal_map):

        prefactor = self.P_500*(self.M_500/3e14*self.h70)**(self.my_params_sz.alpha_p)*self.R_500

        return p_cal_map*prefactor*self.P2Y

    def get_Y_sph(self,x):  #returns spherically integrated Y, assuming self-similarity+alpha_p, in units of Mpc^2. x in units of R_500

        if x == 1:

            I = 0.6145

        elif x == 5:

            I = 1.1037

        return self.Y_500*(self.M_500/3e14*self.h70)**(self.my_params_sz.alpha_p)*I

    def get_Y_cyl(self,x):  #returns cylindrically integrated Y, assuming self-similarity+alpha_p. x in units of R_500.
                            #In units of Mpc^2, i.e., divide by d_A^2 to get true aperture integrated Y

        if x == 1:

            J = 0.7398

        elif x == 5:

            J = 1.1037

        return self.Y_500*(self.M_500/3e14*self.h70)**(self.my_params_sz.alpha_p)*J

    def get_Y_aperture(self,x): #dimensionless (sterad)

        return self.get_Y_cyl(x)/self.d_ad_h**2

    def get_p_cal_int(self,my_params_sz,r): #r is distance in units of R_500

        def p_cal(z):

            x = np.sqrt(z**2+r**2)

            if x < 5:  #truncating at 5 R_500

                p_cal = my_params_sz.P0/((my_params_sz.c_500*x)**my_params_sz.gamma*(1.+(my_params_sz.c_500*x)**my_params_sz.alpha)**((my_params_sz.beta-my_params_sz.gamma)/my_params_sz.alpha))


            else:

                p_cal = 0.

            return p_cal

        #p_int = integrate.quad(p_cal,-np.inf,np.inf,epsabs=1.49e-08/100.)[0]
        p_int = integrate.quad(p_cal,-5.,5.,epsabs=1.49e-08/100.,limit=100)[0]

        return p_int

    def get_t_map(self,pix,exp,theta_misc=[0.,0.],sed=None): #returns t map in units of muK (all freqs)

        y_map = self.get_y_map(pix,theta_misc=theta_misc)

        if sed == None:

            sed = exp.tsz_f_nu

        n_freqs = exp.n_freqs
        t_map = np.zeros((pix.nx,pix.ny,n_freqs))

        for i in range(0,n_freqs):

            t_map[:,:,i] = y_map*sed[i]

        return t_map

    #NOTE: theta_misc is in rad and is [i,j] coord. Positive miscentering in i
    #means "upwards", and in j "leftwards" (i.e., in both cases towards the origin)

    def get_t_map_convolved(self,pix,exp,theta_misc=[0.,0.],theta_cart=None,beam="gaussian",get_nc=False):

        if theta_cart != None:

            theta_misc = maps.get_theta_misc(theta_cart,pix)

        tmap = self.get_t_map(pix,exp,theta_misc=theta_misc)
        tmap_convolved = maps.convolve_tmap_experiment(pix,tmap,exp,beam_type=beam)
        #tmap_convolved = tmap

        if get_nc == True:

            ret = tmap_convolved,tmap

        else:

            ret = tmap_convolved

        return ret

#Delta critical always

class gnfw_tsz:

    def __init__(self,M,z_halo,cosmology,c=1.177,Delta=500.,type="arnaud",path="/Users/user/Desktop/"):

        #Create variables

        self.M = M*1e15
        self.c = c
        self.z_halo = z_halo
        self.cosmology = cosmology
        self.Delta = Delta
        self.type = type
        self.path = path

        #Calculate basic quantities

        const = constants()
        self.rho_c = cosmology.critical_density(self.z_halo).value*1000.*const.mpc**3/const.solar
        self.R_Delta = (3./4.*self.M/self.rho_c/self.Delta/np.pi)**(1./3.)

        self.chi = cosmology.comoving_distance(self.z_halo).value #In Mpc
        self.d_ad = self.chi/(1.+z_halo)

        self.theta_Delta = self.R_Delta/self.d_ad
        self.theta_Delta_arcmin = self.theta_Delta/np.pi*180.*60.
        self.my_params_sz = params_sz(cosmology,c=self.c,type=self.type,M=self.M,z=self.z_halo)

        const = constants()
        self.P2Y = const.mpc*100.*const.sigma_thomson_e/const.mass_e #converts pressure in units of keV cm-3 to dimensionless Compton y

        if self.type == "arnaud":

            self.Ez = self.cosmology.H(self.z_halo).value/self.cosmology.H0.value
            self.h70 = self.cosmology.H0.value/70.
            self.P_500 = 1.65e-3*self.Ez**(8./3.)*(self.M/3e14*self.h70)**(2./3.)*self.h70**2  #in units of keV cm-3
            self.Y_500 = self.P_500*self.R_Delta**3*4.*np.pi/3.*self.P2Y #characteristic Y, in units of Mpc^2

        #elif self.type == "battaglia":

        #    self.rho_c = cosmology.critical_density(self.z_halo).value*1000.*const.mpc**3/const.solar
    #          self.P_Delta = self.G*self.M*self.Delta*self.rho_c*self.cosmology.Omb/self.cosmology.Om0/(2.*self.R_Delta)


    def get_p_cal_map(self,pix,c_500,theta_misc=[0.,0.]):

        #theta_max = np.sqrt((nx*dx)**2+(ny*dy)**2)*0.5*1.1
        theta_max = self.theta_Delta*self.my_params_sz.R_truncation
        theta_vec = np.linspace(0.,theta_max,1000)
        p_cal_vec = np.zeros(len(theta_vec))

        for i in range(0,len(theta_vec)):

            x = theta_vec[i]/self.theta_Delta
            p_cal_vec[i] = self.get_p_cal_int(x)

        theta_map = maps.rmap(pix).get_distance_map_wrt_centre(theta_misc)
        p_cal_map = np.interp(theta_map,theta_vec,p_cal_vec,right=0.)

        return p_cal_map

    def get_y_map(self,pix,theta_misc=[0.,0.]):

        p_cal_map = self.get_p_cal_map(pix,self.c,theta_misc)
        y_map = self.p_cal_to_y(p_cal_map)

        return y_map

    def get_y_norm(self,type="centre"):

        if type == "R_500":

            r = 1.

        elif type == "centre":

            r = 1e-6

        return self.p_cal_to_y(self.get_p_cal_int(r))

    def get_y_at_angle(self,theta): #angle in rad

        return self.p_cal_to_y(self.get_p_cal_int(theta/self.theta_Delta))

    def get_y_map_convolved(self,pix,fwhm_arcmin,theta_misc=[0.,0.]):

        return maps.get_gaussian_convolution(self.get_y_map(pix,theta_misc=theta_misc),fwhm_arcmin,pix)

    def get_y_norm_convolved(self,pix,fwhm_arcmin,y_map=None,theta_misc=[0.,0.]):

        if y_map == None:

            y_map = self.get_y_map_convolved(pix,fwhm_arcmin,theta_misc)

        (theta_x,theta_y) = theta_misc
        x_coord = maps.rmap(pix).get_x_coord_map_wrt_centre(theta_x)[0,:]/self.theta_Delta
        y_coord = maps.rmap(pix).get_y_coord_map_wrt_centre(theta_y)[:,0]/self.theta_Delta

        return interpolate.interp2d(x_coord,y_coord,y_map)(1.,0.)


    def p_cal_to_y(self,p_cal_map):

        if self.type == "arnaud":

            prefactor = self.P_500*(self.M/3e14*self.h70)**(self.my_params_sz.alpha_p)*self.R_Delta

        elif self.type == "battaglia":

            prefactor = 2.61051e-18*self.cosmology.Ob0/self.cosmology.Om0*self.M*self.cosmology.H(self.z_halo).value**2/self.R_Delta*self.R_Delta/1e3

        return p_cal_map*prefactor*self.P2Y

    def get_Y_sph(self,x):  #returns spherically integrated Y, assuming self-similarity+alpha_p, in units of Mpc^2. x in units of R_500

        if x == 1:

            I = 0.6145

        elif x == 5:

            I = 1.1037

        return self.Y_500*(self.M_500/3e14*self.h70)**(self.my_params_sz.alpha_p)*I

    def get_Y_cyl(self,x):  #returns cylindrically integrated Y, assuming self-similarity+alpha_p. x in units of R_500.
                            #In units of Mpc^2, i.e., divide by d_A^2 to get true aperture integrated Y

        if x == 1:

            J = 0.7398

        elif x == 5:

            J = 1.1037

        return self.Y_500*(self.M_500/3e14*self.h70)**(self.my_params_sz.alpha_p)*J

    def get_Y_aperture(self,x): #dimensionless (sterad)

        return self.get_Y_cyl(x)/self.d_ad_h**2

    def pressure_profile(self,x):  #x is distance in units of R_500

        if x < self.my_params_sz.R_truncation:

            if self.type == "arnaud":

                p_cal = self.my_params_sz.P0/((self.my_params_sz.c_500*x)**self.my_params_sz.gamma*(1.+(self.my_params_sz.c_500*x)**self.my_params_sz.alpha)**((self.my_params_sz.beta-self.my_params_sz.gamma)/self.my_params_sz.alpha))

            elif self.type == "battaglia":

                p_cal = self.my_params_sz.P0*(x/self.my_params_sz.xc)**self.my_params_sz.gamma*(1.+(x/self.my_params_sz.xc)**self.my_params_sz.alpha)**(-self.my_params_sz.beta)

        else:

            p_cal = 0.

        return p_cal

    def get_p_cal_int(self,r): #r is distance in units of R_500

        def p_cal(z_los):

            x = np.sqrt(z_los**2+r**2)

            return self.pressure_profile(x)

        p_int = integrate.quad(p_cal,-self.my_params_sz.R_truncation,self.my_params_sz.R_truncation,
        epsabs=1.49e-08/100.,limit=100)[0]

        return p_int

    def get_t_map(self,pix,exp,theta_misc=[0.,0.],eval_type="standard",sed=None): #returns t map in units of muK

        if eval_type == "standard":

            y_map = self.get_y_map(pix,theta_misc=theta_misc)

        elif eval_type == "hankel":

            y_map = self.get_y_map_hankel(pix,theta_misc=theta_misc)

        n_freqs = exp.n_freqs

        if sed is None:

            sed = exp.tsz_f_nu

        elif sed is False:

            sed = np.ones(n_freqs)

        t_map = np.zeros((pix.nx,pix.ny,n_freqs))

        for i in range(0,n_freqs):

            t_map[:,:,i] = y_map*sed[i]

        return t_map

    def get_y_map_hankel(self,pix,theta_misc=[0.,0.]):

        def to_transform(theta):

            return np.vectorize(self.get_p_cal_int)(theta/self.theta_Delta)

        theta_range = [pix.dx/20.,self.theta_Delta*self.my_params_sz.R_truncation*10.]

        rht = maps.RadialFourierTransform()
        rprofs       = to_transform(rht.r)
        lprofs       = rht.real2harm(rprofs)
        ell_vec = rht.l
        rprofs     = rht.harm2real(lprofs)
        r, rprofs    = rht.unpad(rht.r, rprofs)

        theta_map = maps.rmap(pix).get_distance_map_wrt_centre(theta_misc)
        p_cal_map = np.interp(theta_map,r,rprofs,right=0.)
        y_map = self.p_cal_to_y(p_cal_map)

        return y_map

    def get_t_map_convolved_hankel(self,pix,exp,theta_misc=[0.,0.],beam_type="gaussian",get_nc=False,sed=None):

        if sed is None:

            sed = exp.tsz_f_nu

        elif sed is False:

            sed = np.ones(len(exp.tsz_f_nu))

        def to_transform(theta):

            #return np.vectorize(self.pressure_profile)(theta/self.theta_Delta)

            return np.vectorize(self.get_p_cal_int)(theta/self.theta_Delta)

        theta_range = [pix.dx/20.,self.theta_Delta*self.my_params_sz.R_truncation*10.]

        rht = maps.RadialFourierTransform()
        rprofs       = to_transform(rht.r)
        lprofs       = rht.real2harm(rprofs)
        ell_vec = rht.l

        t_map = np.zeros((pix.nx,pix.ny,exp.n_freqs))
        t_map_conv = np.zeros((pix.nx,pix.ny,exp.n_freqs))

        for i in range(0,exp.n_freqs):

            if beam_type == "gaussian":

                beam_fft = maps.get_bl(exp.FWHM[i],ell_vec)

            elif beam_type == "real":

                ell_beam,beam_fft = exp.get_beam(i)
                beam_fft = np.interp(ell_vec,ell_beam,beam_fft)

            rprofs_convolved      = rht.harm2real(lprofs*beam_fft)
            r, rprofs_convolved    = rht.unpad(rht.r, rprofs_convolved)

            rprofs      = rht.harm2real(lprofs)
            r, rprofs    = rht.unpad(rht.r, rprofs)

            theta_map = maps.rmap(pix).get_distance_map_wrt_centre(theta_misc)

            p_cal_map_conv = np.interp(theta_map,r,rprofs_convolved,right=0.)
            p_cal_map = np.interp(theta_map,r,rprofs,right=0.)

            t_map_conv[:,:,i] = self.p_cal_to_y(p_cal_map_conv)*sed[i]
            t_map[:,:,i] = self.p_cal_to_y(p_cal_map)*sed[i]

        if get_nc == True:

            ret = t_map_conv,t_map

        else:

            ret = t_map_conv

        return ret

    #NOTE: theta_misc is in rad and is [i,j] coord. Positive miscentering in i
    #means "upwards", and in j "leftwards" (i.e., in both cases towards the origin)

    def get_t_map_convolved(self,pix,exp,theta_misc=[0.,0.],theta_cart=None,beam="gaussian",get_nc=False,
    eval_type="standard",sed=None):

        if theta_cart != None:

            theta_misc = maps.get_theta_misc(theta_cart,pix)

        if eval_type == "standard":

            tmap = self.get_t_map(pix,exp,theta_misc=theta_misc,eval_type=eval_type,sed=sed)
            tmap_convolved = maps.convolve_tmap_experiment(pix,tmap,exp,beam_type=beam)

        elif eval_type == "hankel":

            tmap_convolved,tmap = self.get_t_map_convolved_hankel(pix,exp,theta_misc=theta_misc,beam_type=beam,get_nc=True,sed=sed)

        if get_nc == True:

            ret = tmap_convolved,tmap

        else:

            ret = tmap_convolved

        return ret

class params_sz: #For above class

    def __init__(self,cosmology,c,z=None,M=None,type="arnaud"):

        if type == "arnaud":

            self.P0 = 8.403*(cosmology.H0.value/70.)**(-1.5)
            self.c_500 = c #1.77 from Arnaud
            self.gamma = 0.3081
            self.alpha = 1.0510
            self.beta = 5.4905
            self.alpha_p = 0.12
            self.R_truncation = 5. #in units of R_500

        elif type == "battaglia":

            self.P0 = param_scal_rel(M,z,18.1,0.154,-0.758)
            self.xc = param_scal_rel(M,z,0.497,-0.00865,0.731)
            self.alpha = 1.
            self.beta = param_scal_rel(M,z,4.35,0.0393,0.415)
            self.gamma = -0.3
            self.R_truncation = 4. #in units of R_200

def param_scal_rel(M,z,A0,alpha_m,alpha_z):

    return A0*(M/1e14)**alpha_m*(1.+z)**alpha_z



class constants:

    def __init__(self):

        self.c_light = 299792458. #in m/s
        self.G = 6.674e-11
        self.solar = 1.98855e30    #Solar mass in kg
        self.mpc = 3.08567758149137e22   #Mpc in m
        self.sigma_thomson_e = 6.6524587158e-25 #in cm2
        self.mass_e = 0.51099895e3 # in keV
        self.k_B = 1.38064852e-23 #in J/K
        self.T_CMB = 2.7255
        self.h = 6.62607004e-34 #in Js



def get_m_500(theta_500_arcmin,z,cosmology): #return M_500 in units of 1e15 solar masses

    theta_500 = theta_500_arcmin/60./180.*np.pi
    R_500 = theta_500*cosmology.comoving_distance(z).value/(1+z)
    const = constants()
    rho_c = cosmology.critical_density(z).value*1000.*const.mpc**3/const.solar
    M_500 = 500.*4.*np.pi/3.*rho_c*R_500**3/1e15

    return M_500

def get_theta_500_arcmin(M_500,z,cosmology): #return M_500 in units of 1e15 solar masses

    M_500 *= 1e15
    const = constants()
    rho_c = cosmology.critical_density(z).value*1000.*const.mpc**3/const.solar
    R_500 = (M_500/(500.*4.*np.pi/3.*rho_c))**(1./3.)
    theta_500 = R_500/(cosmology.comoving_distance(z).value/(1+z))
    theta_500_arcmin = theta_500*60.*180./np.pi

    return theta_500_arcmin

def get_tsz_f_nu(nu,units): #nu in Hz, non relativistic, returns tSZ spectral signature, which times Compton y gives frequency map

    const = constants()
    x = nu*const.h/(const.k_B*const.T_CMB)
    ret = x*(np.exp(x)+1.)/(np.exp(x)-1.) - 4.

    if units == "TCMB":

        ret = ret

    elif units == "muK":

        ret *= const.T_CMB*1e6

    elif units == "K":

        ret *= const.T_CMB

    return ret

def get_t_map(y0,theta_500,theta_misc,cosmology,pix,exp):

    z = 0.2
    M_500 = get_m_500(theta_500,z,cosmology)
    nfw = gnfw_arnaud(M_500,z,cosmology)
    t_tem = nfw.get_t_map_convolved_gaussian(pix,exp,theta_misc=theta_misc)/nfw.get_y_norm()*y0

    return t_tem

def get_t_map_from_catalogue(pix,catalogue,z,cosmology=Planck15,exp=None):

    if exp is None:

        exp = expt.planck_specs()

    catalogue.purge_minus_1()
    n_clus = len(catalogue.q_opt)

    if isinstance(z,float):

        z = np.ones(n_clus)*z

    t_map = np.zeros((pix.nx,pix.ny,6))

    theta_x = catalogue.theta_x
    theta_y = catalogue.theta_y
    theta_500 = catalogue.theta_500
    y0 = catalogue.y0

    for i in range(0,n_clus):

        theta_cart = [theta_x[i],theta_y[i]]
        theta_misc = maps.get_theta_misc(theta_cart,pix)
        nfw = gnfw_arnaud(get_m_500(theta_500[i],z[i],cosmology),z[i],cosmology)
        t_cluster = nfw.get_t_map_convolved_gaussian(pix,exp,theta_misc=theta_misc)
        t_map += t_cluster

    return t_map

def g(x):

    return (np.log(1+x)-x/(1.+x))/x**3

#solve for mass 2

class mass_conversion:

    def __init__(self,cosmology,Delta1,crit1,Delta2,crit2,c1):

        self.cosmology = cosmology
        self.Delta1 = Delta1
        self.Delta2 = Delta2
        self.crit1 = crit1
        self.crit2 = crit2
        self.c1 = c1

    def get_ratio_delta(self,redshift):

        if self.crit1 == self.crit2:

            ratio_delta = np.ones(len(redshift))

        if self.crit1 == "mean" and self.crit2 == "critical":

            ratio_delta = self.cosmology.H(redshift)/(self.cosmology.Om0*(1.+redshift**3))/self.cosmology.H0

        if self.crit1 == "critical" and self.crit2 == "mean":

            ratio_delta = 1./(self.cosmology.H(redshift)/(self.cosmology.Om0*(1.+redshift**3))/self.cosmology.H0)

        ratio_delta = ratio_delta*self.Delta2/self.Delta1
        self.ratio_delta = ratio_delta

        return ratio_delta

    def get_c2(self,redshift):

        ratio_delta = self.get_ratio_delta(redshift)
        c2_vec = np.zeros(len(redshift))

        for i in range(0,len(redshift)):

            if i % 1000 == 0:

                print(i)

            def f2root(x):

                return g(x)-g(self.c1)*ratio_delta[i]

            c2_vec[i] = optimize.root_scalar(f2root,x0=5.,x1=3.).root

        self.c2 = c2_vec

        return c2_vec

    def get_m2(self,M1,redshift,R1=None): #M in units of 1e15 M_Sun

        if R1 is None:

            R1 =  get_R_Delta(M1,redshift,self.cosmology,self.Delta1,self.crit1)

        c2 = self.get_c2(redshift)
        R2 = c2/self.c1*R1
        M2 = self.ratio_delta*(c2/self.c1)**3*M1

        self.R2 = R2
        self.M2 = M2

        return (M2,R2)

def get_R_Delta(M_Delta,z,cosmology,Delta,crit):

    M_Delta = M_Delta*1e15
    const = constants()

    if crit == "critical":

        rho_c = cosmology.critical_density(z).value*1000.*const.mpc**3/const.solar
        R_Delta = (M_Delta/(Delta*4.*np.pi/3.*rho_c))**(1./3.)

    return R_Delta

class scalrel_websky:

    def __init__(self):

        self.cosmology = websky.cosmology_websky()
        self.A0 = 1
        self.B0 = 1
        self.M_pivot = 1
        self.const = constants()

    def get_y0(self,M_500,z):

        E = self.cosmology.H(z)/self.cosmology.H0
        y0 = 10.**self.A0*(M_500/self.M_pivot)**(1.+self.B0)*E**2

        return y0

    def get_theta_500(self,M_500,z,units="arcmin"):

        if units == "arcmin":

            theta_500 = get_theta_500_arcmin(M_500,z,self.cosmology)

        return theta_500

    def get_q(self,M_500,z,theta_vec,sigma_y_vec):

        sigma_y = np.interp(self.get_theta_500(M_500,z,units="arcmin"),theta_vec,sigma_y_vec)
        y0 = self.get_y0(M_500,z)

        return y0/sigma_y


class point_source:

    def __init__(self,pix,exp,beam="gaussian",freqs=None):

        point_source_map = np.zeros((pix.nx,pix.ny,len(freqs)))
        point_source_fft = np.zeros((pix.nx,pix.ny,len(freqs)))

        for i in range(0,len(freqs)):

            if beam =="real":

                if exp.name == "Planck":

                    ell_beam,fft_beam = exp.get_beam(i)

            ell = maps.rmap(pix).get_ell()
            beam_fft = np.interp(ell,ell_beam,fft_beam)
            beam_real = maps.get_ifft(beam_fft,pix).real

            point_source_map[:,:,i] = beam_real
            point_source_fft[:,:,i] = beam_fft

        self.point_source_map = point_source_map
        self.point_source_fft = point_source_fft

class gaussian_signal:

    def __init__(self,pix,A,sigma_arcmin,theta_misc=[0.,0.],n_freqs=1,sed=None):

        self.pix = pix
        self.A = A
        self.sigma_arcmin = sigma_arcmin
        self.sigma = self.sigma_arcmin/60./180.*np.pi
        self.theta_misc = theta_misc
        self.n_freqs = n_freqs
        self.sed = sed

    def get_signal_map(self):

        theta_map = maps.rmap(self.pix).get_distance_map_wrt_centre(self.theta_misc)
        signal_map = self.get_signal(theta_map)*self.A

        ret = np.zeros((self.pix.nx,self.pix.nx,self.n_freqs))

        for i in range(0,self.n_freqs):

            ret[:,:,i] = signal_map*self.sed[i]

        return ret

    def get_t_map_convolved(self,exp,beam="gaussian"):

        tmap = self.get_signal_map()
        n_freqs = exp.n_freqs

        tmap_convolved = maps.convolve_tmap_experiment(self.pix,tmap,exp,beam_type=beam)

        return tmap_convolved

    def get_signal(self,theta):

        return gaussian_1d(theta,0.,self.sigma)

    def get_norm(self):

        return self.get_signal(0.)*self.A



def gaussian_1d(x,mu,sigma):

    y = (x-mu)/sigma

    return np.exp(-0.5*y**2)/(np.sqrt(2.*np.pi*sigma**2))

class cib_model: #Modified blackbody, model used in Websky.

    def __init__(self,path="/Users/user/Desktop/",beta=None,T0=None):

        if beta is None:

            beta = 1.6

        if T0 is None:

            T0 = 20.7

        self.beta = beta
        self.gamma = 1.8
        self.alpha = 0.2
        self.T0 = T0
        self.nu0 = 10000.
        self.nu_pivot = 1#3e9
        self.path = path
        self.moments = {}

    #Input nu is experiment nu (i.e. at z = 0). Returns intensity in T_CMB units.

    def get_Theta(self,nu,z,units="T_CMB",beta=None):

        if beta is not None:

            self.beta = beta

        nup = nu*(1.+z)

        Theta_0 = (nup/self.nu_pivot)**self.beta*planckian(nup,self.T0*(1+z)**self.alpha)

        if units == "T_CMB":

            #Theta = Theta_0*MJysr_to_muK_factor(nu)/1e10
            exp = expt.planck_specs(path=self.path)
            Theta = Theta_0*exp.MJysr_to_muK_websky

        elif units == "SI":

            Theta = Theta_0

        return Theta

    def get_Theta_1d(self,nu,z,exp,units="T_CMB"):

        nup = nu*(1.+z)

        Theta_1_beta = self.get_Theta(nu,z,units=units)*np.log(nup/self.nu_pivot)

        const = constants()
        exponential = np.exp(const.h*nup/( const.k_B*self.T0*(1+z)**self.alpha))

        Theta_1_betaT = -2.*const.h*nup**(3.+self.beta)/(const.c_light**2*(exponential-1.)**2)*exponential*const.h*nup/const.k_B

        if units == "T_CMB":

            Theta_1_betaT = Theta_1_betaT*exp.MJysr_to_muK_websky

        self.Theta_1_betaT = Theta_1_betaT
        self.Theta_1_beta = Theta_1_beta
        self.moments["betaT"] = Theta_1_betaT
        self.moments["beta"] = Theta_1_beta

        return Theta_1_beta,Theta_1_betaT


    def get_sed_exp(self,exp,z,beta=None):

        nu_vec = exp.nu_eff

        return (nu_vec,self.get_Theta(nu_vec,z,beta=beta))

    def get_sed_1d_exp(self,exp,z,units="T_CMB"):

        nu_vec = exp.nu_eff

        Theta_1_beta,Theta_1_betaT = self.get_Theta_1d(nu_vec,z,exp,units=units)

        return Theta_1_beta,Theta_1_betaT

    def get_sed_exp_class_sz(self,exp):

        nu_vec = exp.nu_eff

#    [nu0,I0] = np.load("class_sz_cib_monopole_planck.npy",allow_pickle=True)
        [nu0,I0] = np.load("CIB_monopole_class_sz.npy",allow_pickle=True)
        I0 = I0*MJysr_to_muK_factor(nu0*1e9)
        I0 = np.interp(nu_vec,nu0*1e9,I0)/1e8

        return (nu_vec,I0)

    def get_sed_exp_empirical(self,exp,type="mean"):

        nu_vec = exp.nu_eff

        if type == "mean":

            I0 = np.load("sed_cib_planck_empirical.npy",allow_pickle=True)

        elif type == "cluster":

            I0 = np.load("sed_cib_planck_empirical_clusters.npy",allow_pickle=True)

        elif type == "cluster_only":

    #        sed_empirical_clusters,sed_empirical_random = np.load("sed_cib_planck_empirical_clusters_z03.npy",allow_pickle=True)
        #    sed_empirical_clusters,sed_empirical_random = np.load("sed_cib_planck_empirical_clusters_z03_masked.npy",allow_pickle=True)
            sed_empirical_clusters,sed_empirical_random = np.load("sed_cib_planck_empirical_clusters_z02_masked.npy",allow_pickle=True)
            I0 = sed_empirical_clusters-sed_empirical_random

        return (nu_vec,I0)

#nu in Hz, T in K

def planckian(nu,T):

    const = constants()
    planck = 2.*const.h*nu**3/(const.c_light**2*(np.exp(const.h*nu/(const.k_B*T))-1.))

    return planck

#nu in Hz, T_CMB in K. To go from Jy/sr to T_CMB, dvide by output of this function

def dBnudT(nu,T_CMB=constants().T_CMB):

    const = constants()

    x = const.h*nu/(const.k_B*T_CMB)
    f = x**2*np.exp(x)/(np.exp(x)-1.)**2.

    return 2.*const.k_B*nu**2/const.c_light**2*f

class ps_sz:

    def __init__(self):

        pars = {'h0':0.7, 'obh2':0.025,'och2':0.12,\
            'As':np.exp(3.06)*1e-10,'ns':0.9645,'mnu':0.06,\
            'flag_nu':True, 'flag_tll':False, 'mass_bias':1.5}
        tsz = pysz.tsz_cl()
        ell_arr = np.logspace(1,4.0,20)
        cl_yy, tll = tsz.get_tsz_cl(ell_arr,pars)
        cl = cl_yy[0]+cl_yy[1] #1 halo + 2 halo terms

        self.ell = np.arange(1,4000)
        self.cl = np.interp(np.log(ell),np.log(ell_arr),cl)

#From MJy/sr to muK_CMB. Input frequency in Hz, I in MJy/sr, returns in muK_CMB

def MJysr_to_muK(I,nu):

    const = constants()

    x = const.h*nu/(const.k_B*const.T_CMB)

    return 1.05e3*(np.exp(x)-1.)**2*np.exp(-x)*(nu/1e11)**(-4)*I

def MJysr_to_muK_factor(nu):

    const = constants()

    x = const.h*nu/(const.k_B*const.T_CMB)
    factor = 1.05e3*(np.exp(x)-1.)**2*np.exp(-x)*(nu/1e11)**(-4)

    return factor

class point_source:

    def __init__(self,experiment,beam_type="gaussian"):

        self.experiment = experiment
        self.beam_type = beam_type

    def get_t_map_convolved(self,pix):

        tem = np.zeros((pix.nx,pix.nx,self.experiment.n_freqs))

        for i in range(0,self.experiment.n_freqs):

            if self.beam_type == "gaussian":

                tem[:,:,i] = maps.eval_beam_real_map(pix,self.experiment.FWHM[i])

            elif self.beam_type == "real":

                tem[:,:,i] = maps.eval_real_beam_real_map(pix,self.experiment.get_beam(i))

            tem[:,:,i] = tem[:,:,i]/tem[pix.nx//2,pix.nx//2,i]

        return tem
