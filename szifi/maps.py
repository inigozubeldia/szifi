import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import pylab as pl
from astropy.cosmology import Planck15
from scipy import interpolate
import mmf
import scipy.signal as sp
import pymaster as nmt
import time
import os
import expt
from numba import jit
import scipy.signal as sg
import healpy as hp
import inpaint
import scipy
import scipy.stats as st

class pixel:

    def __init__(self,nx,dx,ny=None,dy=None):

        if ny == None:

            ny = nx

        if dy == None:

            dy = dx

        self.nx = nx
        self.dx = dx
        self.ny = ny
        self.dy = dy

    def get_dx_phys(self,cosmology,z):

        return cosmology.angular_diameter_distance(z).value*self.dx  #in Mpc

    def get_dy_phys(self,cosmology,z):

        return cosmology.angular_diameter_distance(z).value*self.dy  #in Mpc

class rmap:

    def __init__(self,pix,map_value=None,):

        self.nx = pix.nx
        self.dx = pix.dx
        self.ny = pix.ny
        self.dy = pix.dy
        self.pix = pix

        if map_value is None:

            map_value = np.zeros((self.nx,self.ny))

        self.map_value = map_value

    def get_x_coord_map_wrt_centre(self,theta_x=0.):#actually y coordinate, theta_x in radians

        x_coord = (np.arange(0,self.nx)-self.nx*0.5+0.5)*self.dx + theta_x
        x_coord_map = np.transpose(np.tile(x_coord,(self.ny,1)))

        return x_coord_map

    def get_y_coord_map_wrt_centre(self,theta_y=0.):

        y_coord = (np.arange(0,self.ny)-self.ny*0.5+0.5)*self.dy + theta_y
        y_coord_map = np.tile(y_coord,(self.nx,1))

        return y_coord_map

    def get_x_coord_map_wrt_origin(self):#actually y coordinate

        x_coord = (np.arange(0,self.nx)+0.5)*self.dx
        x_coord_map = np.transpose(np.tile(x_coord,(self.ny,1)))

        return x_coord_map

    def get_y_coord_map_wrt_origin(self):

        y_coord = (np.arange(0,self.ny)+0.5)*self.dy
        y_coord_map = np.tile(y_coord,(self.nx,1))

        return y_coord_map

    def get_distance_map_wrt_centre(self,theta_misc=[0.,0.]):

        (theta_x,theta_y) = theta_misc

        return np.sqrt(self.get_x_coord_map_wrt_centre(theta_x)**2+self.get_y_coord_map_wrt_centre(theta_y)**2)

    def get_fft(self):

        return get_fft(self.map_value,self.pix)

    def get_ifft(self):

        return get_ifft(self.map_value,self.pix)

    def get_lxly(self):

        return np.meshgrid(np.fft.fftfreq(self.ny,self.dy)*2.*np.pi,np.fft.fftfreq(self.nx,self.dx)*2.*np.pi)

    def get_ell(self):

        lx, ly = self.get_lxly()

        return np.sqrt(lx**2 + ly**2)

    def convolve_gaussian(self,fwhm_arcmin):

        return get_gaussian_convolution(self.map_value,fwhm_arcmin,self.pix)

    def filter_lrange(self,lrange):

        print(self.get_ell().shape)

        ell =  self.get_ell().reshape(self.pix.nx*self.pix.ny)
        map_filtered = self.map_value.reshape(self.pix.nx*self.pix.ny)

        print(map_filtered.shape)
        #map_filtered[np.where((ell > lrange[1]) & (ell < lrange[0]))] = 0.
        map_filtered.reshape((self.pix.nx,self.pix.ny))

        return map_filtered



class tmap:

    def __init__(self,pix,n_freq,map_value=None):

        self.nx = pix.nx
        self.dx = pix.dx
        self.ny = pix.ny
        self.dy = pix.dy
        self.n_freq = n_freq
        self.pix = pix

        if map_value is None:

            map_value = np.zeros((self.nx,self.ny,self.n_freq))

        self.map_value = map_value

    def get_x_coord_map_wrt_centre(self,theta_x=0.):

        x_coord = (np.arange(0,self.nx)-self.nx*0.5+0.5)*self.dx + theta_x
        x_coord_map = np.tile(x_coord,(self.nx,1))

        return x_coord_map

    def get_y_coord_map_wrt_centre(self,theta_y=0.):

        y_coord = (np.arange(0,self.ny)-self.ny*0.5+0.5)*self.dy + theta_y
        y_coord_map = np.tile(y_coord,(self.ny,1))

        return np.transpose(y_coord_map)

    def get_distance_map_wrt_centre(self,theta_misc=[0.,0.]):

        (theta_x,theta_y) = theta_misc

        return np.sqrt(self.get_x_coord_map_wrt_centre(theta_x)**2+self.get_y_coord_map_wrt_centre(theta_y)**2)

    def get_fft(self):

        fmap_fft = np.zeros((self.n_freq,self.nx,self.ny))

        for i in range(0,self.n_freq):

            fmap_fft[:,:,i] = get_fft(self.map_value[:,:,i],self.pix)

        return fmap_fft

    def get_ifft(self):

        fmap_ifft = np.zeros((self.n_freq,self.nx,self.ny))

        for i in range(0,self.n_freq):

            fmap_ifft[:,:,i] = get_ifft(self.map_value[:,:,i],self.pix)

        return fmap_ifft

    def get_lxly(self):

        return np.meshgrid(np.fft.fftfreq(self.ny,self.dy)*2.*np.pi,np.fft.fftfreq(self.nx,self.dx)*2.*np.pi)

    def get_ell(self):

        lx, ly = self.get_lxly()

        return np.sqrt(lx**2 + ly**2)

    def convolve_gaussian(self,fwhm_arcmin_vec):

        fmap_convolved = np.zeros((self.nx,self.ny,self.n_freq))

        for i in range(0,self.n_freq):

            fmap_convolved[:,:,i] = get_gaussian_convolution(self.map_value[:,:,i],fwhm_arcmin_vec[i],self.pix)

        return fmap_convolved

    def filter_lrange(self,lrange):

        tmap_filtered = np.zeros(self.map_value.shape)

        for i in range(0,self.map_value.shape[2]):

            tmap_filtered[:,:,i] = rmap(self.pix,map_value=self.map_value[:,:,i]).filter_lrange(lrange)

        return tmap_filtered


def get_fft(map_value,pix):

    return np.fft.fft2(map_value)*np.sqrt((pix.dx*pix.dy)/(pix.nx*pix.ny))#same convention as quicklens

def get_ifft(map_value,pix):

    return np.fft.ifft2(map_value)*np.sqrt((pix.nx*pix.ny)/(pix.dx*pix.dy)) #same convention as quicklens

def fftconvolve(map1,map2,pix):

    ret = get_ifft(get_fft(map1,pix)*get_fft(map2,pix),pix).real
    ret2 = np.zeros((pix.nx,pix.ny))

    for i in range(0,pix.nx):

        for j in range(0,pix.ny):

            ret2[i,j] = ret[i-pix.nx//2,j-pix.ny//2]

    return ret2


def get_fft_f(tmap,pix):

    n_freq = tmap.shape[2]
    ret = np.zeros(tmap.shape,dtype=complex)

    for i in range(0,n_freq):

        ret[:,:,i] = get_fft(tmap[:,:,i],pix)

    return ret

def get_ifft_f(tmap,pix):

    n_freq = tmap.shape[2]
    ret = np.zeros(tmap.shape,dtype=complex)

    for i in range(0,n_freq):

        ret[:,:,i] = get_ifft(tmap[:,:,i],pix)

    return ret

def filter_fft(map_fft,pix,ell_filter):

    [lmin,lmax] = ell_filter
    indices = np.where((rmap(pix).get_ell() < lmin) |  (rmap(pix).get_ell() > lmax))
    map_fft[indices] = 0.

    return map_fft

def filter_fft_f(map_fft,pix,ell_filter):

    for i in range(0,map_fft.shape[2]):

        map_fft[:,:,i] = filter_fft(map_fft[:,:,i],pix,ell_filter)

    return map_fft

def filter_tmap(tmap,pix,ell_filter):

    return get_ifft_f(filter_fft_f(get_fft_f(tmap,pix),pix,ell_filter),pix).real

def filter_map(map,pix,ell_filter):

    return get_ifft(filter_fft(get_fft(map,pix),pix,ell_filter),pix).real

def filter_cov(cov,pix,ell_filter):

    [lmin,lmax] = ell_filter
    indices0,indices1 = np.where((rmap(pix).get_ell() < lmin) |  (rmap(pix).get_ell() > lmax))
    cov[indices0,indices1,:,:] = 0.

    return cov

def get_gaussian_convolution(map_value,fwhm_arcmin,pix):

    map_fft = get_fft(map_value,pix)
    ell = rmap(pix).get_ell()
    beam_fft = get_bl(fwhm_arcmin,ell)
    map_fft_convolved = get_ifft(map_fft*beam_fft,pix).real

    return map_fft_convolved

def get_gaussian_convolution_fft(map_fft,fwhm_arcmin,pix):

    ell = rmap(pix).get_ell()
    beam_fft = get_bl(fwhm_arcmin,ell)
    map_fft_convolved = map_fft*beam_fft

    return map_fft_convolved

def get_gaussian_deconvolution(map_value,fwhm_arcmin,pix):

    map_fft = get_fft(map_value,pix)
    ell = rmap(pix).get_ell()
    beam_fft = get_bl(fwhm_arcmin,ell)
    map_fft_convolved = get_ifft(map_fft/beam_fft,pix).real

    return map_fft_convolved

def get_convolution_isotropic(map_value,kernel,pix):

    ell_kernel,fft_kernel = kernel
    map_fft = get_fft(map_value,pix)
    ell = rmap(pix).get_ell()
    beam_fft = np.interp(ell,ell_kernel,fft_kernel)
    map_fft_convolved = get_ifft(map_fft*beam_fft,pix).real

    return map_fft_convolved

def get_convolution_isotropic_fft(map_fft,kernel,pix):

    ell_kernel,fft_kernel = kernel
    ell = rmap(pix).get_ell()
    beam_fft = np.interp(ell,ell_kernel,fft_kernel)
    map_fft_convolved = map_fft*beam_fft

    return map_fft_convolved


def get_deconvolution_isotropic(map_value,kernel,pix):

    ell_kernel,fft_kernel = kernel
    map_fft = get_fft(map_value,pix)
    ell = rmap(pix).get_ell()
    beam_fft = np.interp(ell,ell_kernel,fft_kernel)
    map_fft_convolved = get_ifft(map_fft/beam_fft,pix).real

    return map_fft_convolved

def convolve(map1,map2,pix):

    return get_ifft(get_fft(map1,pix)*get_fft(map2,pix),pix).real

def get_bl(fwhm_arcmin,ell):

    return np.exp(-(fwhm_arcmin*np.pi/180./60.)**2/(16.*np.log(2.))*ell*(ell+1.))

#theta in rad

def get_beam_real(fwhm_arcmin,theta):

    sigma_arcmin = fwhm_arcmin/(2.*np.sqrt(2.*np.log(2)))

    return eval_gaussian(theta,sigma_arcmin/60./180.*np.pi)

def eval_gaussian(x,sigma):

    return st.norm.pdf(x,scale=sigma)

def eval_beam_real_map(pix,fwhm_arcmin):

    theta_map = rmap(pix).get_distance_map_wrt_centre()

    return get_beam_real(fwhm_arcmin,theta_map)

def eval_real_beam_real_map(pix,beam_data):

    ell_map = rmap(pix).get_ell()

    ell_beam,beam_fft = beam_data

    fwhm = expt.planck_specs().FWHM[0]
    beam_gaussian = get_bl(fwhm,ell_map)
    beam_gaussian[0,0] = 0.

    beam_real_space = eval_beam_real_map(pix,fwhm)
    beam_fft_gaussian = get_fft(beam_real_space,pix)

    """
    pl.imshow(beam_gaussian.imag)
    pl.show()
    pl.imshow(beam_fft.imag)
    pl.show()
    pl.imshow(beam_gaussian.real/beam_fft.real)
    pl.show()
    """

    beam_fft = np.interp(ell_map,ell_beam,beam_fft)*np.sign(beam_fft_gaussian)

    return get_ifft(beam_fft,pix)

def get_bl_vec(fwhm_arcmin,lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             = maximum multipole.
    """
    ls = np.arange(0,lmax+1)

    return (ls,get_bl(fwhm_arcmin,ls))

#Beam-deconvolved noise power spectrum

def get_nl(n_lev,fwhm_arcmin,ell):

    return (n_lev*np.pi/180./60.)**2/get_bl(fwhm_arcmin,ell)**2

def get_nl_vec(n_lev,fwhm_arcmin,lmax):

    ls = np.arange(0,lmax+1)

    return (ls,get_nl(n_lev,fwhm_arcmin,ls))


def rfft2_to_fft2(pix,rfft):

    ny = pix.ny
    fft = np.zeros((pix.nx,pix.ny),dtype=complex)
    fft[:,0:(ny//2+1)] = rfft[:,:]
    fft[0,(ny//2+1):]  = np.conj(rfft[0,1:ny//2][::-1])
    fft[1:,(ny//2+1):]  = np.conj(rfft[1:,1:ny//2][::-1,::-1])

    return fft


def nl(noise_uK_arcmin, fwhm_arcmin, lmax):
    """ returns the beam-deconvolved noise power spectrum in units of uK^2 for
         * noise_uK_arcmin = map noise level in uK.arcmin
         * fwhm_arcmin     = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax            = maximum multipole.
    """
    return (noise_uK_arcmin*np.pi/180./60.)**2/bl(fwhm_arcmin,lmax)**2

def get_noise_map(pix,n_lev): #white noise

    sigma_pixel = n_lev/(180.*60./np.pi*pix.dx)

    return get_noise_map_sigma_pix(pix,sigma_pixel)


def get_noise_tmap(pix,n_lev_vec): #generates white noise frequency maps

    tmap = np.zeros((pix.nx,pix.ny,len(n_lev_vec)))

    for i in range(0,len(n_lev_vec)):

        tmap[:,:,i] = get_noise_map(pix,n_lev_vec[i])

    return tmap

def get_noise_tmap_Planck(pix,ptf_flag=False):

    noise = get_noise_tmap(pix,expt.planck_specs().noise_levels)

    if ptf_flag == True:

        exp = expt.planck_specs()

        for i in range(0,exp.n_freqs):

            noise[:,:,i] = get_convolution_isotropic(noise[:,:,i],exp.get_ptf(),pix)

    return noise

def get_noise_tmap_exp(pix,exp,ptf_flag=False):

    noise = get_noise_tmap(pix,exp.noise_levels)

    if ptf_flag == True:

        for i in range(0,exp.n_freqs):

            noise[:,:,i] = get_convolution_isotropic(noise[:,:,i],exp.get_ptf(),pix)

    return noise

def get_deconvolved_noise_map(pix,n_lev,fwhm_arcmin,lmax):

    ell,spec = get_nl_vec(n_lev,fwhm_arcmin,lmax)

    return get_realisation_from_spec_real(pix,ell,spec)

def get_noise_map_sigma_pix(pix,sigma_pixel):

    return np.random.standard_normal((pix.nx,pix.ny))*sigma_pixel

def get_nlev_from_sigma_pixel(pix,sigma_pixel):

    return sigma_pixel*(180.*60./np.pi*np.sqrt(pix.dx*pix.dy))


def get_cl_noise(ell,my_sz_data):

    return np.interp(ell,my_sz_data.l_eff,my_sz_data.cl_all)

def cl_to_map(ell,cl,pix,lmax=None):

    ell_map = rmap(pix).get_ell()
    ret = np.interp(ell_map,ell,cl)

    if lmax != None:

        indices = np.where(ell_map > lmax)
        ret[indices] = 0.
        print(np.where(ret==0.))

    return ret

def cl_binned_to_map(ell_bin,cl_bin,pix):

    ell = np.arange(ell_bin[0],ell_bin[-1])
    cl = np.interp(ell,ell_bin,cl_bin)
    ell_map = rmap(pix).get_ell()

    return np.interp(ell_map,ell,cl)

def get_noise_cl_map(pix):

    data = sz_data()
    ell_map = rmap(pix).get_ell()

    return get_cl_noise(ell_map,data)

def get_noise_cl_map_gaussian(pix,n_lev,fwhm_arcmin):

    ell_map = rmap(pix).get_ell()
    noise = get_nl(n_lev,fwhm_arcmin,ell_map)

    return noise

def get_cross_spectrum(pix,map1,map2=None,ell=None): #deprecated

    if ell == None:

        ell = rmap(pix).get_ell()

    if map2.all() == None:

        map2 = map1

    map1 = get_fft(map1,pix).flatten()
    map2 = get_fft(map2,pix).flatten()
    ell = ell.flatten()
    idx_sort = np.argsort(ell)
    map1_sorted = map1[idx_sort]
    map2_sorted = map2[idx_sort]
    ell_sorted = ell[idx_sort]
    ell_new, idx_start, count = np.unique(ell_sorted, return_counts=True, return_index=True)
    cross_spec = np.zeros(len(ell_new))

    for i in range(0,len(ell_new)):

        mapps1 = map1_sorted[idx_start[i]:idx_start[i]+count[i]]
        mapps2 = map2_sorted[idx_start[i]:idx_start[i]+count[i]]
        cross_spec[i] = np.mean(mapps1*np.conjugate(mapps2)).real

    return (ell_new,cross_spec)



class ps_mask:

    def __init__(self,pix,n_source,r_source_arcmin):

        self.pix = pix
        self.n_source = n_source
        self.r_source_arcmin = r_source_arcmin

    def get_source_coords(self): #with respect to the centre of the map

        source_coords = np.random.rand(self.n_source,2)-0.5
        source_coords[:,0] = source_coords[:,0]*self.pix.nx*self.pix.dx
        source_coords[:,1] = source_coords[:,1]*self.pix.ny*self.pix.dy

        return source_coords

    def get_mask_map(self,source_coords=None):

        if source_coords is None:

            source_coords = self.get_source_coords()

        mask = np.ones((self.pix.nx,self.pix.ny))

        for i in range(0,self.n_source):

            distances = rmap(self.pix).get_distance_map_wrt_centre(theta_misc=source_coords[i,:])
            r_source = self.r_source_arcmin/60./180.*np.pi
            indices = np.where(distances <= r_source)
            mask[indices] = 0.

        return mask

    def get_mask_map_t(self,n_freq,source_coords=None):

        mask = np.zeros((self.pix.nx,self.pix.ny,n_freq))
        mask_map = self.get_mask_map(source_coords=source_coords)

        for i in range(n_freq):

            mask[:,:,i] = mask_map

        return mask

def clone_map_freq(map1,n_freq):

    ret = np.zeros((map1.shape[0],map1.shape[1],n_freq))

    for i in range(0,n_freq):

        ret[:,:,i] = map1

    return ret

def get_mask_apod(pix,alpha=0.1):

    line1 = sp.tukey(pix.nx,alpha)
    line2 = sp.tukey(pix.ny,alpha)

    return np.outer(line1,line2)

def get_mask_apod_freq(pix,n_freq=6,alpha=0.1):

    mask = np.zeros((pix.nx,pix.ny,n_freq))

    for i in range(0,n_freq):

        mask[:,:,i] = get_mask_apod(pix,alpha=alpha)

    return mask

def get_mask_square(nx,buffer,ny=None):

    if ny is None:

        ny = nx

    mask = np.zeros((nx,ny))
    mask[buffer:nx-buffer,buffer:ny-buffer] = 1.

    return mask

def get_theta_cart(theta_misc,pix):

    theta_x_misc,theta_y_misc = theta_misc
    theta_x = pix.nx/2*pix.dx - theta_y_misc
    theta_y = pix.nx/2*pix.dx + theta_x_misc
    theta_cart = [theta_x,theta_y]

    return theta_cart

def get_theta_misc(theta_cart,pix):

    [theta_x,theta_y] = theta_cart
    theta_y_misc = pix.nx/2*pix.dx - theta_x
    theta_x_misc = -pix.nx/2*pix.dx + theta_y
    theta_misc = [theta_x_misc,theta_y_misc]

    return theta_misc

def get_ij_from_theta(theta_x,theta_y,pix):

    j = np.floor(theta_x/pix.dx).astype("int")
    i = (pix.ny-np.ceil(theta_y/pix.dy)).astype("int")

    return i,j

def get_theta_from_ij(i,j,pix):

    theta_x = j*pix.dx
    theta_y = (pix.ny-i)*pix.dy

    return theta_x,theta_y


@jit
def get_buffered_mask(pix,mask_input,buffer_arcmin):

    mask_output = np.ones(mask_input.shape)
    buffer_pix = int(round(buffer_arcmin/60./180.*np.pi/pix.dx))+1

    for i in range(0,mask_input.shape[0]):

        for j in range(0,mask_input.shape[1]):


            theta_x = j*pix.dy
            theta_y = (pix.nx-i)*pix.dx
            theta_cart = [theta_x,theta_y]
            distances = rmap(pix).get_distance_map_wrt_centre(theta_misc=get_theta_misc(theta_cart,pix))

            imin = i-buffer_pix
            imax = i+buffer_pix
            jmin = j-buffer_pix
            jmax = j+buffer_pix

            if imin < 0:

                imin = 0

            if imax >= pix.nx:

                imax = pix.nx-1

            if jmin < 0:

                jmin = 0

            if jmax >= pix.nx:

                jmax = pix.nx-1

            distances_select = distances[imin:imax,jmin:jmax]
            mask_select = mask_input[imin:imax,jmin:jmax]

            indices = np.where(distances_select <= buffer_arcmin/60./180.*np.pi)
            mask_select = mask_select[indices]

            if 0. in mask_select:

                mask_output[i,j] = 0.

    return mask_output


def get_buffered_mask_fft(pix,mask_input,buffer_arcmin):

    buffer_rad = buffer_arcmin/60./180.*np.pi
    buffer_pix = int(round(buffer_arcmin/60./180.*np.pi/pix.dx))+1

    nx_padded = pix.nx+4*buffer_pix
    ny_padded = pix.ny+4*buffer_pix

    mask_input_padded = np.ones((nx_padded,ny_padded))
    mask_input_padded[2*buffer_pix:nx_padded-2*buffer_pix,2*buffer_pix:ny_padded-2*buffer_pix] = mask_input

    pix_padded = pixel(nx_padded,pix.dx,ny=ny_padded,dy=pix.dy)

    kernel = np.zeros((pix_padded.nx,pix_padded.ny))
    distances = rmap(pix_padded).get_distance_map_wrt_centre()
    indices = np.where(distances <= buffer_rad)
    kernel[indices] = 1.
    kernel_area = np.sum(kernel)

    mask_convolved = np.around(sg.fftconvolve(mask_input_padded,kernel,mode='same')/kernel_area,decimals=4)[2*buffer_pix:nx_padded-2*buffer_pix,2*buffer_pix:ny_padded-2*buffer_pix]

    mask_output = np.ones(mask_input.shape)
    mask_output[np.where(mask_convolved != 1.)] = 0.

    return mask_output

#apotype: C1, C2 and Smooth

def get_apodised_mask(pix,mask_input,apotype="C1",aposcale=0.2):

    mask_input[0,:] = 0.
    mask_input[:,0] = 0.
    mask_input[-1,:] = 0.
    mask_input[:,-1] = 0.

    return nmt.mask_apodization_flat(mask_input,pix.nx*pix.dx,pix.ny*pix.dy,aposcale,apotype=apotype)

def get_fsky_criterion_mask(pix,mask_select,nside_tile,criterion=0.3):

    area_tile = hp.pixelfunc.nside2pixarea(nside_tile)
    area_select = np.sum(mask_select)*pix.dx*pix.dy
    frac = area_select/area_tile
    print("Tile frac",frac)

    if frac < criterion:

        mask_select = np.zeros((pix.nx,pix.ny))

    return mask_select

def mask_sigma(map,sigma,inpaint_flag=False,n_inpaint=100):

    std = np.std(map)
    mean = np.mean(map)
    indices = np.where(np.abs(map)>sigma*std+mean)
    map[indices] = 0.

    if inpaint_flag == True:

        mask = np.ones(map.shape)
        mask[indices] = 0.
        map = inpaint.diffusive_inpaint(map,mask,n_inpaint)

    return map

def mask_sigma_freq(tmap,sigma,inpaint_flag=False):

    ret = np.zeros(tmap.shape)

    for i in range(0,tmap.shape[2]):

        ret[:,:,i] = mask_sigma(tmap[:,:,i],sigma=sigma,inpaint_flag=inpaint_flag)

    return ret

def extract_value_at_catalogue(tsz,catalogue_tile,pix):

    i,j = get_ij_from_theta(catalogue_tile.theta_x,catalogue_tile.theta_y,pix)

    return tsz[i,j]

def get_binned_mean_map(map,pix,bins,fft_flag=False):

    map_real = rmap(pix,map)

    if fft_flag == False:

        map_fft = map_real.get_fft()

    else:

        map_fft = map

    ell = map_real.get_ell()

    binned_map = np.zeros(len(bins)-1)
    bins_centres = np.zeros(len(binned_map))

    for i in range(0,len(bins_centres)):

        indices = np.where((ell > bins[i]) & (ell < bins[i+1]))
        selected_map = np.mean(map_fft[indices]).real
        binned_map[i] = selected_map
        bins_centres[i] = 0.5*(bins[i] + bins[i+1])

    return bins_centres,binned_map

def convolve_tmap_experiment(pix,tmap,exp,freqs=None,beam_type="gaussian"):

    n_freqs = tmap.shape[2]
    freqs = np.arange(n_freqs)
    tmap_convolved = np.zeros(tmap.shape)

    for i in range(0,n_freqs):

        if beam_type == "gaussian":

            tmap_convolved[:,:,i] = get_gaussian_convolution(tmap[:,:,i],exp.FWHM[freqs[i]],pix)

        elif beam_type == "real":

            tmap_convolved[:,:,i] = get_convolution_isotropic(tmap[:,:,i],exp.get_beam(freqs[i]),pix)

    return tmap_convolved

def convolve_tmap_fft_experiment(pix,tmap_fft,exp,freqs=None,beam_type="gaussian"):

    n_freqs = tmap_fft.shape[2]
    freqs = np.arange(n_freqs)
    tmap_convolved_fft = np.zeros(tmap_fft.shape)

    for i in range(0,n_freqs):

        if beam_type == "gaussian":

            tmap_convolved_fft[:,:,i] = get_gaussian_convolution_fft(tmap_fft[:,:,i],exp.FWHM[freqs[i]],pix)

        elif beam_type == "real":

            tmap_convolved_fft[:,:,i] = get_convolution_isotropic_fft(tmap_fft[:,:,i],exp.get_beam(freqs[i]),pix)

    return tmap_convolved_fft


def get_tmap_times_fvec(tmap,fvec):

    tmap_ret = np.zeros(tmap.shape,dtype=complex)

    for i in range(0,tmap.shape[2]):

        tmap_ret[:,:,i] = tmap[:,:,i]*fvec[i]

    return tmap_ret

def get_tmap_from_map(map,n_freq):

    tmap = np.zeros((map.shape[0],map.shape[1],n_freq))

    for i in range(0,n_freq):

        tmap[:,:,i] = map

    return tmap


def select_freqs(tmap,freqs):

    return tmap[:,:,freqs]

def get_tmap_with_sed(pix,tmap,base_freq,exp,sed,beam_type="gaussian"):

    sed = sed/sed[base_freq]
    ret = np.zeros(tmap.shape)

    deconv_freq = base_freq

    if beam_type == "gaussian":

        spatial_map = get_gaussian_deconvolution(tmap[:,:,deconv_freq],exp.FWHM[deconv_freq],pix)

    elif beam_type == "real":

        spatial_map = get_deconvolution_isotropic(tmap[:,:,deconv_freq],exp.get_beam(deconv_freq),pix)

    for i in range(0,tmap.shape[2]):

        ret[:,:,i] = spatial_map*sed[i]

    ret = convolve_tmap_experiment(pix,ret,exp,freqs=np.arange(len(sed)),beam_type=beam_type)

    return ret

def get_hankel_transform(theta_range,function,n=512,pad=128):

    lrange = [1/theta_range[1], 1/theta_range[0]]

    logl1, logl2 = np.log(lrange)
    logl0        = (logl2+logl1)/2
    dlog    = (logl2-logl1)/n
    i0           = (n+1)/2+pad
    l       = np.exp(logl0 + (np.arange(1,n+2*pad+1)-i0)*dlog)
    r       = 1/l[::-1]
    rprof = function(r)
    lprof = 2*np.pi*scipy.fft.fht(rprof*r,dlog,0)/l

    res = [l,lprof]
    l,lprof = tuple([arr[...,pad:-pad] for arr in res])

    return l,r,lprof,dlog

class RadialFourierTransform:

    def __init__(self, lrange=None, rrange=None, n=512//4, pad=256//4):
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


def profile_to_tform_hankel(profile_fun, lmin=0.1, lmax=1e7, n=512, pad=256):
    """Transform a radial profile given by the function profile_fun(r) to
    sperical harmonic coefficients b(l) using a Hankel transform. This approach
    is good at handling cuspy distributions due to using logarithmically spaced
    points. n points from 10**logrange[0] to 10**logrange[1] will be used.
    Returns l, bl. l will not be equi-spaced, so you may want to interpolate
    the results. Note that unlike other similar functions in this module and
    the curvedsky module, this function uses the flat sky approximation, so
    it should only be used for profiles up to a few degrees in size."""
    rht   = RadialFourierTransform(lrange=[lmin,lmax], n=n, pad=pad)
    lprof = rht.real2harm(profile_fun)
    return rht.unpad(rht.l, lprof)


def bin_radially(pix,map,bins_edges=None,nbins=None,theta_misc=[0.,0.]):

    theta_map = rmap(pix).get_distance_map_wrt_centre(theta_misc)

    if bins_edges is None:

        bins_edges = np.linspace(0,np.max(theta_map),nbins)

    bins = np.zeros(len(bins_edges)-1)
    binned_map = np.zeros(len(bins_edges)-1)

    theta_flattened = theta_map.flatten()
    map_flattened = map.flatten()

    for i in range(0,len(bins_edges)-1):

        bins[i] = 0.5*(bins_edges[i+1]+bins_edges[i])
        indices = np.where((theta_flattened < bins_edges[i+1]) & (theta_flattened >= bins_edges[i]))[0]
        binned_map[i] = np.mean(map_flattened[indices])

    return bins,binned_map
