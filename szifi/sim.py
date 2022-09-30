import numpy as np
from astropy.cosmology import Planck15
from scipy import integrate
import spec
import maps
import model
import mmf
import expt
import time
import healpy as hp

def get_realisation_from_spec_fft(pix,ell,cl):

    my_map = maps.rmap(pix)

    nx = pix.nx
    ny = pix.ny
    map_fft = (np.random.standard_normal((nx,ny//2+1))+1.j*np.random.standard_normal((nx,ny//2+1)))/np.sqrt(2.)
    map_fft[0,0] = np.sqrt(2.)*np.real(map_fft[0,0])
    map_fft[nx//2+1:,0] = np.conj(map_fft[1:nx//2,0][::-1])

    map_fft = maps.rfft2_to_fft2(pix,map_fft)
    map_fft[:,:] *= np.interp(my_map.get_ell(),ell,np.sqrt(cl),right=0)

    return map_fft

def get_realisation_from_spec_real(pix,ell,cl):

    return maps.get_ifft(get_realisation_from_spec_fft(pix,ell,cl),pix)


def get_cmb_realisation(pix): #CMB realisation in muK

    ell,cl = spec.get_camb_cltt()

    return get_realisation_from_spec_real(pix,ell,cl).real

#Returns CMB realisation convolved with Gaussian Planck beams in muK (all channels)

def get_cmb_realisation_gaussian(pix,n_freq=None,non_periodic=True,exp=None,beam="gaussian"):

    n_freq = exp.n_freqs

    if non_periodic == True:

        fac = 2
        nx = pix.nx*fac
        ny = pix.ny*fac
        pix = maps.pixel(nx,pix.dx,ny=ny)

    cmb_realisation = np.zeros((pix.nx,pix.ny,n_freq))

    cmb = get_cmb_realisation(pix)

    for i in range(0,n_freq):

        if beam == "gaussian":

            cmb_realisation[:,:,i] = maps.get_gaussian_convolution(cmb,exp.FWHM[i],pix)

        elif beam == "real":

            cmb_realisation[:,:,i] = maps.get_convolution_isotropic(cmb,exp.get_beam(i),pix)


    if non_periodic == True:

        cmb_realisation = cmb_realisation[nx-nx//fac:nx+nx//fac,ny-ny//fac:ny+ny//fac,:]

    return cmb_realisation



def get_gaussian_realisation(pix,n_freq=None,non_periodic=True,exp=None,beam="gaussian",field="cmb",z=0.):

    n_freq = exp.n_freqs

    if non_periodic == True:

        fac = 2
        nx = pix.nx*fac
        ny = pix.ny*fac
        pix = maps.pixel(nx,pix.dx,ny=ny)

    realisation = np.zeros((pix.nx,pix.ny,n_freq))

    if field == "cmb":

        ell,cl = spec.get_camb_cltt()
        cl_nu = np.repeat(cl[:,np.newaxis],2,axis=1)

    elif field == "cib":

        ell = np.arange(2,100000)
        cross_spec = spec.cross_spec_theory(ell,exp,cmb_flag=False,cib_flag=True,tsz_flag=False,
        noise_flag=False,tsz_cib_flag=False)

        cl_nu = cross_spec[:,i]


    realisation_unconvolved = get_realisation_from_spec_real(pix,ell,cl).real

    for i in range(0,n_freq):

        if beam == "gaussian":

            realisation[:,:,i] = maps.get_gaussian_convolution(cmb,exp.FWHM[i],pix)*sed[i]

        elif beam == "real":

            realisation[:,:,i] = maps.get_convolution_isotropic(cmb,exp.get_beam(i),pix)*sed[i]


    if non_periodic == True:

        cmb_realisation = cmb_realisation[nx-nx//fac:nx+nx//fac,ny-ny//fac:ny+ny//fac,:]

    return cmb_realisation

class simulated_map:

    def __init__(self,pix,pixel_id=0,cosmology=Planck15,n_clus=None,
    generate=True,non_periodic=True,cmb_flag=True,noise_flag=True,cluster_flag=True,
    catalogue=None,concentration=1.177,beam="gaussian",
    locations_clusters="random",pixel_buffer_cluster=0.,n_freqs=6,exp=None):

        t_obs = None
        cmb = None
        noise = None
        cluster_sz = None

        if generate == True:

            t_obs = np.zeros((pix.nx,pix.ny,n_freqs))

            if noise_flag == True:

                noise = maps.get_noise_tmap_exp(pix,exp)
                t_obs += noise

            if cmb_flag == True:

                cmb = get_cmb_realisation_gaussian(pix,non_periodic=non_periodic,exp=exp,beam=beam)
                t_obs += cmb

            if cluster_flag == True:

                if catalogue is not None:

                    cluster_tsz,catalogue = get_cluster_tsz_map(pix,catalogue,pixel_id=pixel_id,cosmology=cosmology,
                    concentration=concentration,pixel_buffer=pixel_buffer_cluster,locations=locations_clusters,beam=beam)
                    t_obs += cluster_tsz

        self.t_obs = t_obs
        self.cmb = cmb
        self.noise = noise
        self.cluster_sz = cluster_sz
        self.true_catalogue = catalogue
        self.n_freqs = 6.

    def save(self,name):

        np.save(name + "_map.npy",(self.t_obs,self.cmb,self.noise,self.cluster_sz,self.true_catalogue))

    def load(self,name):

        self.t_obs,self.cmb,self.noise,self.cluster_sz,self.true_catalogue = np.load(name + "_map.npy",allow_pickle=True)


def get_cluster_tsz_map(pix,catalogue,nfreq=6,cosmology=Planck15,concentration=1.177,pixel_buffer=0.,locations="random",pixel_id=0,beam="gaussian"):

    m_500_vec = catalogue.m_500
    z_vec = catalogue.z
    n_clus = len(m_500_vec)

    t_tsz = np.zeros((pix.nx,pix.ny,nfreq))

    if n_clus > 0:

        theta_500_vec = model.get_theta_500_arcmin(m_500_vec,z_vec,cosmology)
        pixel_ids = np.ones(n_clus)*pixel_id

        y0_vec = np.zeros(n_clus)

        if locations == "random":

            theta_x_vec = np.random.rand(n_clus)*(pix.nx-pixel_buffer)*pix.dx+pixel_buffer*pix.dx
            theta_y_vec = np.random.rand(n_clus)*(pix.nx-pixel_buffer)*pix.dx+pixel_buffer*pix.dx

            for i in range(0,n_clus):

                theta_misc = maps.get_theta_misc([theta_x_vec[i],theta_y_vec[i]],pix)
                nfw = model.gnfw_arnaud(model.get_m_500(theta_500_vec[i],z_vec[i],cosmology),z_vec[i],cosmology)
                t_cluster = nfw.get_t_map_convolved(pix,"Planck",theta_misc=theta_misc,beam=beam)
                t_tsz += t_cluster
                y0_vec[i] = nfw.get_y_norm()

        catalogue = mmf.cluster_catalogue(y0=y0_vec,
        theta_500=theta_500_vec,theta_x=theta_x_vec,theta_y=theta_y_vec,pixel_ids=pixel_ids)

    return t_tsz,catalogue

def get_cluster_tsz_map_from_cat(pix,catalogue,nfreq=6,cosmology=Planck15,concentration=1.177,beam="gaussian"):

    m_500_vec = catalogue.m_500
    z_vec = catalogue.z
    n_clus = len(m_500_vec)

    t_tsz = np.zeros((pix.nx,pix.ny,nfreq))

    if n_clus > 0:

        y0_vec = np.zeros(n_clus)

        for i in range(0,n_clus):

            theta_misc = maps.get_theta_misc([catalogue.theta_x[i],catalogue.theta_y[i]],pix)
            nfw = model.gnfw_arnaud(m_500_vec[i]/1e15,z_vec[i],cosmology)
            t_cluster = nfw.get_t_map_convolved(pix,"Planck",theta_misc=theta_misc,beam=beam)
            t_tsz += t_cluster
            y0_vec[i] = nfw.get_y_norm()

    catalogue.y0 = y0_vec

    return t_tsz

def sample_on_sphere(n_samples):

    vec = np.random.randn(3,n_samples)
    vec /= np.linalg.norm(vec, axis=0)
    lon,lat = hp.pixelfunc.vec2ang(vec,lonlat=True)

    return lon,lat

def get_white_noise_hpmap(nside,n_lev):

    pixel_area = hp.pixelfunc.nside2pixarea(nside)
    sigma_pixel = n_lev/np.sqrt(pixel_area)/(180.*60./np.pi)
    n_pixels = hp.pixelfunc.nside2npix(nside)

    return np.random.standard_normal(n_pixels)*sigma_pixel

"""
class sim_full_sky_maps:

    def __init__(nside=4096,exp="Planck",freqs=[0,1,2,3,4,5],cmb_flag=True,cib_flag=True,
    tsz_flag=True,ksz_flag=True,gdust_flag=True,synchro_flag=True,freefree_flag=True,radio_flag=True):

        npix = hp.nside2npix(nside)
        map = np.zeros(npix)

        if cib_flag == True:
"""
