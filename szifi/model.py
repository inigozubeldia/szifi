import numpy as np
from scipy import integrate, interpolate
import scipy.optimize as optimize
from szifi import maps
import astropy.cosmology as cp

#Delta critical always

class gnfw:

    def __init__(self,M,z_halo,cosmology,c=1.177,Delta=500.,type="arnaud",R_truncation=5.):

        #Create variables

        self.M = M*1e15
        self.c = c
        self.z_halo = z_halo
        self.cosmology = cosmology
        self.Delta = Delta
        self.type = type
        self.R_truncation = R_truncation

        #Calculate basic quantities

        const = constants()
        self.rho_c = cosmology.critical_density(self.z_halo).value*1000.*const.mpc**3/const.solar
        self.R_Delta = (3./4.*self.M/self.rho_c/self.Delta/np.pi)**(1./3.)
        #self.chi = cosmology.comoving_distance(self.z_halo).value #In Mpc
        self.d_ad = cosmology.angular_diameter_distance(self.z_halo).value

        self.theta_Delta = self.R_Delta/self.d_ad
        self.theta_Delta_arcmin = self.theta_Delta/np.pi*180.*60.
        self.my_params_sz = params_sz(cosmology,c=self.c,type=self.type,M=self.M,z=self.z_halo)
        self.my_params_sz.R_truncation = self.R_truncation

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

        from .maps import rmap

        theta_map = rmap(pix).get_distance_map_wrt_centre(theta_misc)
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

            r = 1e-8#1e-6

        p_cal_int = self.get_p_cal_int(r)

        return self.p_cal_to_y(p_cal_int)

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

        return self.Y_500*(self.M/3e14*self.h70)**(self.my_params_sz.alpha_p)*I

    def get_Y_cyl(self,x):  #returns cylindrically integrated Y, assuming self-similarity+alpha_p. x in units of R_500.
                            #In units of Mpc^2, i.e., divide by d_A^2 to get true aperture integrated Y

        if x == 1:

            J = 0.7398

        elif x == 5:

            J = 1.1037

        #return self.Y_500*(self.M/3e14*self.h70)**(self.my_params_sz.alpha_p)*J

        return 2.925e-5*J/self.h70/self.d_ad**2*self.Ez**(2./3.)*(self.M/3e14*self.h70)**(5./3.+self.my_params_sz.alpha_p)


    def get_Y_aperture(self,x): #dimensionless (sterad) it is only valid for the Arnaud cosmology (H0=70,Om0=0.3), otherwise use get_Y_aperture_numerical

        return self.get_Y_cyl(x)

    def get_Y_aperture_numerical(self,x): #x in units of theta_500

        theta_max = self.theta_Delta*x
        theta_vec = np.linspace(0.,theta_max,1000)
        p_cal_vec = np.zeros(len(theta_vec))

        for i in range(0,len(theta_vec)):

            x = theta_vec[i]/self.theta_Delta
            p_cal_vec[i] = self.get_p_cal_int(x)

        y_vec = self.p_cal_to_y(p_cal_vec)

        self.y_vec = y_vec
        self.theta_vec = theta_vec

        y_vec[0] = 0.
        Y_aperture = integrate.simps(y_vec*2.*np.pi*theta_vec,theta_vec)

        return Y_aperture


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

            sed = exp.tsz_sed

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

        rht = radialFourierTransform()
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

        theta_vec,t_vec_conv,t_vec = self.get_t_vec_convolved_hankel(pix,exp,beam_type=beam_type,get_nc=True,sed=sed)

        theta_map = maps.rmap(pix).get_distance_map_wrt_centre(theta_misc)

        t_map = np.zeros((pix.nx,pix.ny,exp.n_freqs))
        t_map_conv = np.zeros((pix.nx,pix.ny,exp.n_freqs))

        for i in range(0,exp.n_freqs):

            t_map_conv[:,:,i] = np.interp(theta_map,theta_vec,t_vec_conv[:,i],right=0.)
            t_map[:,:,i] = np.interp(theta_map,theta_vec,t_vec[:,i],right=0.)

        if get_nc == True:

            ret = t_map_conv,t_map

        else:

            ret = t_map_conv

        return ret

    def get_t_vec_convolved_hankel(self,pix,exp,beam_type="gaussian",get_nc=False,sed=None):

        if sed is None:

            sed = exp.tsz_sed

        elif sed is False:

            sed = np.ones(len(exp.tsz_sed))

        def to_transform(theta):

            return np.vectorize(self.get_p_cal_int)(theta/self.theta_Delta)

        theta_range = [pix.dx/10.,self.theta_Delta*self.my_params_sz.R_truncation*20.]

        rht = radialFourierTransform(rrange=theta_range)
        rprofs = to_transform(rht.r)
        lprofs = rht.real2harm(rprofs)
        ell_vec = rht.l

        r_temp, rprofs_temp = rht.unpad(rht.r,rprofs)

        t_vec = np.zeros((len(r_temp),exp.n_freqs))
        t_vec_conv = np.zeros((len(r_temp),exp.n_freqs))

        for i in range(0,exp.n_freqs):

            if beam_type == "gaussian":

                beam_fft = maps.get_bl(exp.FWHM[i],ell_vec)

            elif beam_type == "real":

                ell_beam,beam_fft = exp.get_beam(i)
                beam_fft = np.interp(ell_vec,ell_beam,beam_fft)

            rprofs_convolved = rht.harm2real(lprofs*beam_fft)
            r, rprofs_convolved = rht.unpad(rht.r,rprofs_convolved)

            rprofs = rht.harm2real(lprofs)
            r, rprofs = rht.unpad(rht.r,rprofs)

            t_vec_conv[:,i] = self.p_cal_to_y(rprofs_convolved)*sed[i]
            t_vec[:,i] = self.p_cal_to_y(rprofs)*sed[i]

        if get_nc == True:

            ret = r,t_vec_conv,t_vec

        else:

            ret = r,t_vec_conv

        return ret

    #NOTE: theta_misc is in rad and is [i,j] coord. Positive miscentering in i
    #means "upwards", and in j "leftwards" (i.e., in both cases towards the origin)

    def get_t_map_convolved(self,pix,exp,theta_misc=[0.,0.],theta_cart=None,beam="gaussian",get_nc=False,
    eval_type="standard",sed=None):

        if theta_cart != None:

            from .maps import get_theta_misc
            theta_misc = get_theta_misc(theta_cart,pix)

        if eval_type == "standard":

            tmap = self.get_t_map(pix,exp,theta_misc=theta_misc,eval_type=eval_type,sed=sed)
            from .maps import convolve_tmap_experiment
            tmap_convolved = convolve_tmap_experiment(pix,tmap,exp,beam_type=beam)

        elif eval_type == "hankel":

            tmap_convolved,tmap = self.get_t_map_convolved_hankel(pix,exp,theta_misc=theta_misc,beam_type=beam,get_nc=True,sed=sed)
            #tmap_convolved,tmap = self.get_t_map_convolved_hankel2(pix,exp,theta_misc=theta_misc,beam_type=beam,get_nc=True,sed=sed)

        if get_nc == True:

            ret = tmap_convolved,tmap

        else:

            ret = tmap_convolved

        return ret

class params_sz: #For above class

    def __init__(self,cosmology,c,z=None,M=None,type="arnaud"):

        if type == "arnaud":

            self.P0 = 8.403*(cosmology.H0.value/70.)**(-1.5)
            self.c_500 = c #1.177 from Arnaud
            self.gamma = 0.3081
            self.alpha = 1.0510
            self.beta = 5.4905
            self.alpha_p = 0.12
            self.R_truncation = 5. #in units of R_500

        elif type == "battaglia":

            def param_scal_rel(M,z,A0,alpha_m,alpha_z):

                return A0*(M/1e14)**alpha_m*(1.+z)**alpha_z

            self.P0 = param_scal_rel(M,z,18.1,0.154,-0.758)
            self.xc = param_scal_rel(M,z,0.497,-0.00865,0.731)
            self.alpha = 1.
            self.beta = param_scal_rel(M,z,4.35,0.0393,0.415)
            self.gamma = -0.3
            self.R_truncation = 4. #in units of R_200

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
    R_500 = theta_500*cosmology.angular_diameter_distance(z).value
    const = constants()
    rho_c = cosmology.critical_density(z).value*1000.*const.mpc**3/const.solar
    M_500 = 500.*4.*np.pi/3.*rho_c*R_500**3/1e15

    return M_500

def get_theta_500_arcmin(M_500,z,cosmology): #return M_500 in units of 1e15 solar masses

    M_500 *= 1e15
    const = constants()
    rho_c = cosmology.critical_density(z).value*1000.*const.mpc**3/const.solar
    R_500 = (M_500/(500.*4.*np.pi/3.*rho_c))**(1./3.)
    theta_500 = R_500/cosmology.angular_diameter_distance(z).value
    theta_500_arcmin = theta_500*60.*180./np.pi

    return theta_500_arcmin


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


class point_source:

    def __init__(self,experiment,beam_type="gaussian"):

        self.experiment = experiment
        self.beam_type = beam_type

    def get_t_map_convolved(self,pix):

        tem = np.zeros((pix.nx,pix.nx,self.experiment.n_freqs))

        for i in range(0,self.experiment.n_freqs):

            if self.beam_type == "gaussian":

                tem[:,:,i] = maps.eval_beam_real_map(pix,self.experiment.FWHM[i])

            tem[:,:,i] = tem[:,:,i]/tem[pix.nx//2,pix.nx//2,i]

        return tem

class cosmological_model:

    def __init__(self,params_szifi):

        name = params_szifi["cosmology"]

        if name == "Planck15":
            self.cosmology = cp.Planck15

        elif name == "Websky":
            self.Ob0 = 0.049
            self.Oc0 = 0.261
            self.Om0 = self.Ob0 + self.Oc0
            self.h      = 0.68
            self.ns     = 0.965
            self.sigma8 = 0.81
            self.Neff = 3.046
            self.m_nu = [0.06, 0, 0]
            self.Tcmb0 = 2.7255
            self.cosmology = cp.FlatLambdaCDM(Om0=self.Om0,H0=self.h*100.,Ob0=self.Ob0, Tcmb0=self.Tcmb0, Neff=self.Neff, m_nu=self.m_nu)
            self.As = 2.079522e-09

        elif name == "cosmocnc":

            import cosmocnc

            cosmology_tool = params_szifi["cosmology_tool"]
            cosmo_params = cosmocnc.cosmo_params_default
            cosmology_model = cosmocnc.cosmology_model(cosmo_params=cosmo_params,cosmology_tool=cosmology_tool,amplitude_parameter="sigma_8")
            self.cosmology = cosmology_model.background_cosmology

#Taken from pixell

class radialFourierTransform:

    def __init__(self, lrange=None, rrange=None, n=512, pad=256):
        """Construct an object for transforming between radially
        symmetric profiles in real-space and fourier space using a
        fast Hankel transform. Aside from being fast, this is also
        good for representing both cuspy and very extended profiles
        due to the logarithmically spaced sample points the fast
        Hankel transform uses. A cost of this is that the user can't
        freely choose the sample points. Instead one passes the
        multipole range or radial range of interest as well as the
        number of points to use.

        The function currently assumes two dimensions with flat geometry.
        That means the function is only approximate for spherical
        geometries, and will only be accurate up to a few degrees
        in these cases.

        Arguments:
        * lrange = [lmin, lmax]: The multipole range to use. Defaults
          to [0.01, 1e6] if no rrange is given.
        * rrange = [rmin, rmax]: The radius range to use if lrange is
        	not specified, in radians. Example values: [1e-7,10].
        	Since we don't use spherical geometry r is not limited to 2 pi.
        * n: The number of logarithmically equi-spaced points to use
        	in the given range. Default: 512. The Hankel transform usually
        	doesn't need many points for good accuracy, and can suffer if
        	too many points are used.
        * pad: How many extra points to pad by on each side of the range.
          Padding is useful to get good accuracy in a Hankel transform.
          The transforms this function does will return padded output,
        	which can be unpadded using the unpad method. Default: 256
        """
        if lrange is None and rrange is None: lrange = [0.1, 1e7]
        if lrange is None: lrange = [1/rrange[1], 1/rrange[0]]
        logl1, logl2 = np.log(lrange)
        logl0        = (logl2+logl1)/2
        self.dlog    = (logl2-logl1)/n
        i0           = (n+1)/2+pad
        self.l       = np.exp(logl0 + (np.arange(1,n+2*pad+1)-i0)*self.dlog)
        self.r       = 1/self.l[::-1]
        self.pad     = pad

    def real2harm(self, rprof):
        """Perform a forward (real -> harmonic) transform, taking us from the
        provided real-space radial profile rprof(r) to a harmonic-space profile
        lprof(l). rprof can take two forms:
        1. A function rprof(r) that can be called to evalute the profile at
           arbitrary points.
        2. An array rprof[self.r] that provides the profile evaluated at the
           points given by this object's .r member.
        The transform is done along the last axis of the profile.
        Returns lprof[self.l]. This includes padding, which can be removed
        using self.unpad"""
        import scipy.fft
        try: rprof = rprof(self.r)
        except TypeError: pass
        lprof = 2*np.pi*scipy.fft.fht(rprof*self.r, self.dlog, 0)/self.l
        return lprof

    def harm2real(self, lprof):
        """Perform a backward (harmonic -> real) transform, taking us from the
        provided harmonic-space radial profile lprof(l) to a real-space profile
        rprof(r). lprof can take two forms:
        1. A function lprof(l) that can be called to evalute the profile at
           arbitrary points.
        2. An array lprof[self.l] that provides the profile evaluated at the
           points given by this object's .l member.
        The transform is done along the last axis of the profile.
        Returns rprof[self.r]. This includes padding, which can be removed
        using self.unpad"""
        import scipy.fft
        try: lprof = lprof(self.l)
        except TypeError: pass
        rprof = scipy.fft.ifht(lprof/(2*np.pi)*self.l, self.dlog, 0)/self.r
        return rprof

    def unpad(self, *arrs):
        """Remove the padding from arrays used by this object. The
        values in the padded areas of the output of the transform have
        unreliable values, but they're not cropped automatically to
        allow for round-trip transforms. Example:
        	r = unpad(r_padded)
        	r, l, vals = unpad(r_padded, l_padded, vals_padded)"""
        if self.pad == 0: res = arrs
        else: res = tuple([arr[...,self.pad:-self.pad] for arr in arrs])
        return res[0] if len(arrs) == 1 else res

#Taken from pixell

def profile_to_tform_hankel(profile_fun, lmin=0.1, lmax=1e7, n=512, pad=256):
    """Transform a radial profile given by the function profile_fun(r) to
    sperical harmonic coefficients b(l) using a Hankel transform. This approach
    is good at handling cuspy distributions due to using logarithmically spaced
    points. n points from 10**logrange[0] to 10**logrange[1] will be used.
    Returns l, bl. l will not be equi-spaced, so you may want to interpolate
    the results. Note that unlike other similar functions in this module and
    the curvedsky module, this function uses the flat sky approximation, so
    it should only be used for profiles up to a few degrees in size."""
    rht   = radialFourierTransform(lrange=[lmin,lmax], n=n, pad=pad)
    lprof = rht.real2harm(profile_fun)
    return rht.unpad(rht.l, lprof)

def get_bl(fwhm_arcmin,ell):

    return np.exp(-(fwhm_arcmin*np.pi/180./60.)**2/(16.*np.log(2.))*ell*(ell+1.))

def y0_to_T(y0,z,E_z,D_A,h70):

    alpha_szifi = 1.12
    bias_sz = 0.62
    A_szifi = -4.3054
    I = 0.06728373215772082 #see test_scaling_relation_int.py

    prefactor_M_to_y0 = 10.**(A_szifi)*E_z**2*(1./3.*h70)**alpha_szifi/np.sqrt(h70)
    prefactor_M_500_to_theta = 6.997*(h70)**(-2./3.)*(1./3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

    M_sr = (y0/prefactor_M_to_y0)**(1./alpha_szifi)/bias_sz
    theta_500_sr = prefactor_M_500_to_theta*(bias_sz*M_sr)**(1./3.)

    Y_500 = y0*theta_500_sr**2*np.pi*I

    arcmin_to_rad = np.pi/60./180.

    Y_500_Mpc = Y_500*D_A**2*arcmin_to_rad**2

    T = Y_500_to_T(Y_500_Mpc,z,E_z)

    return T

def y0_to_Y_500(y0,theta_500):

    I = 0.06728373215772082 #see test_scaling_relation_int.py
    Y_500 = y0*theta_500**2*np.pi*I

    return Y_500

def y0_to_Y_500_Mpc(y0,theta_500,D_A):

    arcmin_to_rad = np.pi/60./180.
    Y_500 =  y0_to_Y_500(y0,theta_500)
    Y_500_Mpc = Y_500*D_A**2*arcmin_to_rad**2

    return Y_500_Mpc


def Y_500_to_T(Y_500_Mpc,z,E_z):

    z_rsz = [0.,0.25,0.5,1.,1.5]
    A_rsz = [3.123,2.839,2.584,2.113,1.765]
    B_rsz = [0.364,0.369,0.363,0.361,0.360]

    A_rsz_vec = np.interp(z,z_rsz,A_rsz)
    B_rsz_vec = np.interp(z,z_rsz,B_rsz)

    T = E_z**(2./5.)*A_rsz_vec*(Y_500_Mpc/1e-5)**B_rsz_vec

    return T
