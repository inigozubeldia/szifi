import numpy as np
from numba import jit
from scipy import interpolate
import scipy.signal as sp
import pymaster as nmt
import scipy.signal as sg
import healpy as hp
import scipy
import scipy.stats as st

#Functions for handling maps

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
    
    #This works for CAR maps too

    def get_lxly(self):

        lx,ly = np.meshgrid(np.fft.fftfreq(self.nx,self.dx)*2.*np.pi,np.fft.fftfreq(self.ny,self.dy)*2.*np.pi)
        lx = lx.transpose()
        ly = ly.transpose()

        return lx,ly

    def get_ell(self):

        lx, ly = self.get_lxly()

        return np.sqrt(lx**2 + ly**2)

    def convolve_gaussian(self,fwhm_arcmin):

        return get_gaussian_convolution(self.map_value,fwhm_arcmin,self.pix)

    def filter_lrange(self,lrange):

        #ell =  self.get_ell().reshape(self.pix.nx*self.pix.ny)
        map_filtered = self.map_value.reshape(self.pix.nx*self.pix.ny)

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

# def fftconvolve(map1, map2, pix=None):
#     return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(map1) * np.fft.fft2(map2)).real)

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

def filter_fft(map_fft,pix,ell_filter, indices_filter=None):

    if indices_filter is None:

        [lmin,lmax] = ell_filter
        indices_filter = np.where((rmap(pix).get_ell() < lmin) |  (rmap(pix).get_ell() > lmax))

    map_fft[indices_filter] = 0.

    return map_fft

def filter_fft_f(map_fft,pix,ell_filter,indices_filter=None):

    for i in range(0,map_fft.shape[2]):

        map_fft[:,:,i] = filter_fft(map_fft[:,:,i],pix,ell_filter,indices_filter=indices_filter)

    return map_fft

def filter_tmap(tmap,pix,ell_filter,indices_filter=None):
    dtype = tmap.dtype
    return np.asarray(get_ifft_f(filter_fft_f(get_fft_f(tmap,pix),pix,ell_filter,indices_filter=indices_filter),pix).real, dtype=dtype)

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

def get_bl_vec(fwhm_arcmin,lmax):

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

def resample_fft(d,n,axes=None):

        """Resample numpy array d via fourier-reshaping. Requires periodic data.
        n indicates the desired output lengths of the axes that are to be
        resampled. By default the last len(n) axes are resampled, but this
        can be controlled via the axes argument.
        This function borrowed from Sigurd Naess' pixell,
        Copyright (c) 2018-2021, Members of the Simons Observatory Collaboration"""

        d = np.asanyarray(d)
        # Compute output lengths from factors if necessary
        n = np.atleast_1d(n)

        if axes is None: axes = np.arange(-len(n),0)

        else: axes = np.atleast_1d(axes)

        if len(n) == 1: n = np.repeat(n, len(axes))

        else: assert len(n) == len(axes)

        assert len(n) <= d.ndim

        # Nothing to do?

        if np.all(d.shape[-len(n):] == n): return d

        # Use the simple version if we can. It has lower memory overhead

        if d.ndim == 2 and len(n) == 1 and (axes[0] == 1 or axes[0] == -1):

                return resample_fft_simple(d, n[0])

        # Perform the fourier transform
        fd = np.fft.fftn(d, axes=axes)
        # Frequencies are 0 1 2 ... N/2 (-N)/2 (-N)/2+1 .. -1
        # Ex 0* 1 2* -1 for n=4 and 0* 1 2 -2 -1 for n=5
        # To upgrade,   insert (n_new-n_old) zeros after n_old/2
        # To downgrade, remove (n_old-n_new) values after n_new/2
        # The idea is simple, but arbitrary dimensionality makes it
        # complicated.
        norm = 1.0

        for ax, nnew in zip(axes, n):

                ax %= d.ndim
                nold = d.shape[ax]
                dn   = nnew-nold

                if dn > 0:

                        padvals = np.zeros(fd.shape[:ax]+(dn,)+fd.shape[ax+1:],fd.dtype)
                        spre  = tuple([slice(None)]*ax+[slice(0,nold//2)]+[slice(None)]*(fd.ndim-ax-1))
                        spost = tuple([slice(None)]*ax+[slice(nold//2,None)]+[slice(None)]*(fd.ndim-ax-1))
                        fd = np.concatenate([fd[spre],padvals,fd[spost]],axis=ax)

                elif dn < 0:

                        spre  = tuple([slice(None)]*ax+[slice(0,nnew//2)]+[slice(None)]*(fd.ndim-ax-1))
                        spost = tuple([slice(None)]*ax+[slice(nnew//2-dn,None)]+[slice(None)]*(fd.ndim-ax-1))
                        fd = np.concatenate([fd[spre],fd[spost]],axis=ax)

                norm *= float(nnew)/nold
        # And transform back
        res  = np.fft.ifftn(fd, axes=axes, norm='backward')

        del fd

        res *= norm
        
        return res if np.issubdtype(d.dtype, np.complexfloating) else res.real

def resample_fft_simple(d,n,ngroup=100):

        """Resample 2d numpy array d via fourier-reshaping along
        last axis.
        This function borrowed from Sigurd Naess' pixell,
        Copyright (c) 2018-2021, Members of the Simons Observatory Collaboration"""

        nold = d.shape[1]

        if n == nold: return d

        res  = np.zeros([d.shape[0],n],dtype=d.dtype)
        dn   = n-nold

        for di in range(0, d.shape[0], ngroup):

                fd = np.fft.fftn(d[di:di+ngroup])

                if n < nold:

                        fd = np.concatenate([fd[:,:n//2],fd[:,n//2-dn:]],1)

                else:

                        fd = np.concatenate([fd[:,:nold//2],np.zeros([len(fd),n-nold],fd.dtype),fd[:,nold//2:]],-1)

                res[di:di+ngroup] = np.fft.ifftn(fd, norm='backward').real

        del fd

        res *= float(n)/nold

        return res

def get_newshape_lmax1d(shape, lmax1d, dx_rad, powerOfTwo=False):

    """Get new shape for an array set by 1d-lmax (ie actual lmax will be sqrt(lmax_x^2 + lmax_y^2))"""

    if len(shape) > 2:

        raise ValueError("Expected 2-tuple shape")

    newdx = np.pi / lmax1d

    if newdx <= dx_rad:

        return shape

    extent = np.array(shape) * dx_rad
    new_shape = np.ceil(extent / newdx).astype(int)

    if powerOfTwo:

        new_shape = (2**np.ceil(np.log2(new_shape))).astype(int)

    return tuple(new_shape)

def degrade_map(arr, new_shape, deg_axes=[0,1]):

    """Degrade a map by setting the new shape"""

    return resample_fft(arr, new_shape, deg_axes)

def degrade_pix(pix, new_shape):

    """
    Degrade a pixel object
    pix: szifi.maps.pixel object
    new_shape: tuple, directly set new shape instead of using lmax
    """

    if new_shape is None:

        return pix

    nx, ny = new_shape
    new_dx = pix.dx * (pix.nx / nx)
    new_dy = pix.dy * (pix.ny / ny)
    deg_pix = pixel(nx, new_dx, ny, new_dy)

    return deg_pix

def reshape_ell_matrix(arr, new_shape, axes=[0,1]):

    """
    Zero fill or cut arr in the given axes to reach new_shape
    This splits arr into quadrants and adds zeros (or removes vals) in the central cross; this is how to expand inv_cov or anything shaped like ell
    arr: np.ndarr
    new_shape: tuple, new shape of *the axes to be expanded only*
    This function adapted from pixell.resample.resample_fft
    """

    if np.all(np.array([arr.shape[ax] for ax in axes]) == np.array(new_shape)):

        return arr

    for ax, nnew in zip(axes, new_shape):

        ax %= arr.ndim
        nold = arr.shape[ax]
        dn = nnew - nold

        if dn < 0:

            spre  = tuple([slice(None)]*ax+[slice(0,nnew//2)]+[slice(None)]*(arr.ndim-ax-1))
            spost = tuple([slice(None)]*ax+[slice(nnew//2-dn,None)]+[slice(None)]*(arr.ndim-ax-1))
            arr = np.concatenate([arr[spre], arr[spost]], axis=ax)

        elif dn == 0:

            continue

        else:

            padvals = np.zeros(arr.shape[:ax]+(dn,)+arr.shape[ax+1:],arr.dtype)
            spre  = tuple([slice(None)]*ax+[slice(0,nold//2)]+[slice(None)]*(arr.ndim-ax-1))
            spost = tuple([slice(None)]*ax+[slice(nold//2,None)]+[slice(None)]*(arr.ndim-ax-1))
            arr = np.concatenate([arr[spre],padvals,arr[spost]],axis=ax)

    return arr

def get_noise_map(pix,n_lev): #white noise

    sigma_pixel = n_lev/(180.*60./np.pi*pix.dx)

    return get_noise_map_sigma_pix(pix,sigma_pixel)


def get_noise_tmap(pix,n_lev_vec): #generates white noise frequency maps

    tmap = np.zeros((pix.nx,pix.ny,len(n_lev_vec)))

    for i in range(0,len(n_lev_vec)):

        tmap[:,:,i] = get_noise_map(pix,n_lev_vec[i])

    return tmap

def get_noise_tmap_exp(pix,exp,ptf_flag=False):

    noise = get_noise_tmap(pix,exp.noise_levels)

    if ptf_flag == True:

        for i in range(0,exp.n_freqs):

            noise[:,:,i] = get_convolution_isotropic(noise[:,:,i],exp.get_ptf(),pix)

    return noise

# def get_deconvolved_noise_map(pix,n_lev,fwhm_arcmin,lmax):

#     ell,spec = get_nl_vec(n_lev,fwhm_arcmin,lmax)

#     return get_realisation_from_spec_real(pix,ell,spec)

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

    return ret

def cl_binned_to_map(ell_bin,cl_bin,pix):

    ell = np.arange(ell_bin[0],ell_bin[-1])
    cl = np.interp(ell,ell_bin,cl_bin)
    ell_map = rmap(pix).get_ell()

    return np.interp(ell_map,ell,cl)

# def get_noise_cl_map(pix):

#     data = sz_data()
#     ell_map = rmap(pix).get_ell()

#     return get_cl_noise(ell_map,data)

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

def clone_map_freq(map1,n_freq):

    ret = np.zeros((map1.shape[0],map1.shape[1],n_freq))

    for i in range(0,n_freq):

        ret[:,:,i] = map1

    return ret
def multiply_t(map1, map2):
    """Add an axis to 'map1' and multiply it by 'map2'"""
    if map1.ndim > map2.ndim:
        map1, map2 = map2, map1
    imap = np.expand_dims(map1, map1.ndim) # Add a len-1 axis for compatible shapes
    return imap * map2

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


def get_buffered_mask(pix,mask_input,buffer_arcmin,type="fft",tile_type="healpix",wcs=None):

    buffer_rad = buffer_arcmin/60./180.*np.pi
    buffer_pix_x = int(round(buffer_arcmin/60./180.*np.pi/pix.dx))+1
    buffer_pix_y = int(round(buffer_arcmin/60./180.*np.pi/pix.dy))+1

    nx_padded = pix.nx+4*buffer_pix_x
    ny_padded = pix.ny+4*buffer_pix_y

    mask_input_padded = np.ones((nx_padded,ny_padded))
    mask_input_padded[2*buffer_pix_x:nx_padded-2*buffer_pix_x,2*buffer_pix_y:ny_padded-2*buffer_pix_y] = mask_input
    pix_padded = pixel(nx_padded,pix.dx,ny=ny_padded,dy=pix.dy)

    kernel = np.zeros((pix_padded.nx,pix_padded.ny))

    if tile_type == "healpix":

        distances = rmap(pix_padded).get_distance_map_wrt_centre()

    elif tile_type == "car":

        from pixell import enmap, utils

        mask_input_padded = enmap.enmap(mask_input_padded,wcs)
        position_map = enmap.empty(mask_input_padded.shape,wcs)
        pos = position_map.posmap()  # shape: (2, Ny, Nx)
        dec_map, ra_map = pos
        pix_center = [(mask_input_padded.shape[-2] - 1) / 2, (mask_input_padded.shape[-1] - 1) / 2]
        dec0, ra0 = enmap.pix2sky(mask_input_padded.shape,wcs,pix_center)

        distances = utils.angdist([ra0,dec0], [ra_map,dec_map]) 

    indices = np.where(distances <= buffer_rad)
    kernel[indices] = 1.
    kernel_area = np.sum(kernel)

    mask_convolved = np.around(sg.fftconvolve(mask_input_padded,kernel,mode='same')/kernel_area,decimals=4)[2*buffer_pix_x:nx_padded-2*buffer_pix_x,2*buffer_pix_y:ny_padded-2*buffer_pix_y]

    mask_output = np.ones(mask_input.shape)
    mask_output[np.where(mask_convolved != 1.)] = 0.

    return mask_output

def get_buffer_region(pix, mask, buffer_arcmin):

    buffered_mask = get_buffered_mask(pix, mask, buffer_arcmin, type='fft')
    ibuffered_mask = get_buffered_mask(pix, 1-mask, buffer_arcmin, type='fft')
    buffer_region = 1 - buffered_mask - ibuffered_mask

    return buffer_region

#apotype: C1, C2 and Smooth

def get_apodised_mask(pix,mask_input2,apotype="C1",aposcale=0.2):

    mask_input = np.copy(mask_input2)

    mask_input[0,:] = 0.
    mask_input[:,0] = 0.
    mask_input[-1,:] = 0.
    mask_input[:,-1] = 0.

    return nmt.mask_apodization_flat(mask_input,pix.nx*pix.dx,pix.ny*pix.dy,aposcale,apotype=apotype)

def get_fsky_criterion_mask(pix,mask_select,nside_tile,criterion=0.3,tile_type="healpix"):

    if tile_type == "healpix":

        area_tile = hp.pixelfunc.nside2pixarea(nside_tile)

    elif tile_type == "car": #note that this is not the physical area, but that's okay here 

        area_tile = pix.nx*pix.ny*pix.dx*pix.dy

    area_select = np.sum(mask_select)*pix.dx*pix.dy
    frac = area_select/area_tile

    if frac < criterion:

        mask_select = np.zeros((pix.nx,pix.ny))

    return mask_select

class ps_mask:

    def __init__(self,pix,n_source,r_source_arcmin,tile_type="healpix",wcs=None):

        self.pix = pix
        self.n_source = n_source
        self.r_source_arcmin = r_source_arcmin
        self.tile_type = tile_type
        self.wcs = wcs
        self.r_source = self.r_source_arcmin/60./180.*np.pi

    def get_mask_map(self,source_coords=None):

        mask = np.ones((self.pix.nx,self.pix.ny))

        if self.tile_type == "car":

            from pixell import enmap, utils

        for i in range(0,self.n_source):

            if self.tile_type == "healpix":

                distances = rmap(self.pix).get_distance_map_wrt_centre(theta_misc=source_coords[i,:])

            elif self.tile_type == "car":

                position_map = enmap.empty(mask.shape,self.wcs)
                pos = position_map.posmap()  # shape: (2, Ny, Nx)
                dec_map, ra_map = pos
                distances = utils.angdist([source_coords[i,0],source_coords[i,1]], [ra_map,dec_map]) 

            indices = np.where(distances <= self.r_source)
            mask[indices] = 0.

        return mask

    def get_mask_map_t(self,n_freq,source_coords=None):

        mask = np.zeros((self.pix.nx,self.pix.ny,n_freq))
        mask_map = self.get_mask_map(source_coords=source_coords)

        for i in range(n_freq):

            mask[:,:,i] = mask_map

        return mask

def mask_sigma(map,sigma,inpaint_flag=False,n_inpaint=100):

    std = np.std(map)
    mean = np.mean(map)
    indices = np.where(np.abs(map)>sigma*std+mean)
    map[indices] = 0.

    if inpaint_flag == True:

        mask = np.ones(map.shape)
        mask[indices] = 0.
        map = diffusive_inpaint(map,mask,n_inpaint)

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

def convolve_tmap_experiment(pix,tmap,exp,freqs=None,beam_type="gaussian",tile_type="healpix",wcs=None):

    n_freqs = tmap.shape[2]
    freqs = np.arange(n_freqs)
    tmap_convolved = np.zeros(tmap.shape)

    for i in range(0,n_freqs):

        if tile_type == "healpix":

            if beam_type == "gaussian":

                tmap_convolved[:,:,i] = get_gaussian_convolution(tmap[:,:,i],exp.FWHM[freqs[i]],pix)

            elif beam_type == "real":

                tmap_convolved[:,:,i] = get_convolution_isotropic(tmap[:,:,i],exp.get_beam(freqs[i]),pix)

        elif tile_type == "car":

            from pixell import enmap, enplot

            if beam_type == "gaussian":

                map_freq = enmap.enmap(tmap[:,:,i],wcs)
                sigma = exp.FWHM[freqs[i]]/(2.*np.sqrt(2.*np.log(2)))/60./180.*np.pi
                tmap_convolved[:,:,i] = enmap.smooth_gauss(map_freq, sigma)

            elif beam_type == "real":

                tmap_convolved = None #todo

    return tmap_convolved

def convolve_tmap_fft_experiment(pix,tmap_fft,exp,freqs=None,beam_type="gaussian"):

    n_freqs = tmap_fft.shape[2]

    if freqs is None:

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

def bin_radially(pix,map,bins_edges=None,nbins=None,theta_misc=[0.,0.],return_n_pixels=False):

    theta_map = rmap(pix).get_distance_map_wrt_centre(theta_misc)

    if bins_edges is None:

        bins_edges = np.linspace(0,np.max(theta_map),nbins)

    bins = np.zeros(len(bins_edges)-1)
    binned_map = np.zeros(len(bins_edges)-1)
    n_pixels = np.zeros(len(bins_edges)-1)

    theta_flattened = theta_map.flatten()
    map_flattened = map.flatten()

    for i in range(0,len(bins_edges)-1):

        bins[i] = 0.5*(bins_edges[i+1]+bins_edges[i])
        indices = np.where((theta_flattened < bins_edges[i+1]) & (theta_flattened > bins_edges[i]))#[0]
        binned_map[i] = np.mean(map_flattened[indices])
        n_pixels[i] = len(indices[0])

    if return_n_pixels == True:

        ret = bins,binned_map,n_pixels

    else:

        ret = bins,binned_map

    return ret

def get_integrated_map_r(pix,map,theta_misc=[0.,0.],equality_type="less_or_equal"):

    integrated_map = np.zeros(map.shape)
    theta_map = rmap(pix).get_distance_map_wrt_centre(theta_misc)

    thetas = np.unique(theta_map.flatten())

    for i in range(0,len(thetas)):

        indices = np.where(theta_map == thetas[i])

        if equality_type == "less_or_equal":

            indices_integrated = np.where(theta_map <= thetas[i])

        elif equality_type == "less":

            indices_integrated = np.where(theta_map < thetas[i])

        integrated_map[indices] = np.mean(map[indices_integrated])

    return integrated_map

def repixelise(image,repixelise_factor,type="average"):

    repixelise_factor = int(repixelise_factor)
    nx = image.shape[0]
    nx_rep = nx//repixelise_factor
    image_repixelised = np.zeros((nx_rep,nx_rep))

    for i in range(0,image_repixelised.shape[0]):

        for j in range(0,image_repixelised.shape[1]):

            if type == "average": #repixelised pixels have to be an exact division of each original pixel

                image_repixelised[i,j] = np.mean(image[repixelise_factor*i:(repixelise_factor*i)+repixelise_factor,repixelise_factor*j:(repixelise_factor*j)+repixelise_factor])

            elif type == "central": #the original pixels have to be split in an odd number of pixels in each direction

                shift = (repixelise_factor-1)//2
                image_repixelised[i,j] = image[repixelise_factor*i+shift,repixelise_factor*j+shift]

    return image_repixelised


def bin_harmonic_map(map,pix,ell_bins_edges):

    ell_map = rmap(pix).get_ell()
    ell_bins_centres = (ell_bins_edges[0:-1]+ell_bins_edges[1:])*0.5
    binned_map = np.zeros(len(ell_bins_centres))

    for i in range(0,len(binned_map)):

        indices = np.where((ell_map.flatten() > ell_bins_edges[i]) & (ell_map.flatten() < ell_bins_edges[i+1]))

        binned_map[i] = np.mean(map.flatten()[indices])

    return ell_bins_centres,binned_map

def paint_profile(pix,theta_rad_vec,profile_vec):

    distance_map = rmap(pix).get_distance_map_wrt_centre()

    return np.interp(distance_map,theta_rad_vec,profile_vec)

def repixelise_map(pix_in,map_in,pix_out,method="cubic"):

    rmap_in = rmap(pix_in)
    theta_x = rmap_in.get_x_coord_map_wrt_centre()[:,0]
    theta_y = rmap_in.get_y_coord_map_wrt_centre()[0,:]
    interp = interpolate.RegularGridInterpolator((theta_x,theta_y),map_in,bounds_error=False,fill_value=0.,method=method)

    rmap_out = rmap(pix_out)
    theta_x = rmap_out.get_x_coord_map_wrt_centre()
    theta_y = rmap_out.get_y_coord_map_wrt_centre()
    map_out = interp((theta_x,theta_y))

    return map_out

def get_map_convolved_fft(map_fft_original,pix,freqs,beam_type,mask,lrange,exp):

    a_map_fft = convolve_tmap_fft_experiment(pix,map_fft_original,exp,freqs=freqs,beam_type=beam_type)
    a_map_fft = get_fft_f(get_ifft_f(a_map_fft,pix)*mask,pix)
    a_map_fft = filter_fft_f(a_map_fft,pix,lrange)

    return a_map_fft

def diffusive_inpaint_freq(tmap,mask,n_inpaint):

    ret = np.zeros(tmap.shape, dtype=tmap.dtype)

    for i in range(0,tmap.shape[2]):

        ret[:,:,i] = diffusive_inpaint(tmap[:,:,i],mask,n_inpaint)

    return ret

@jit(nopython=False)
def diffusive_inpaint(image,mask,n_inpaint):

    nx = len(image)

    if np.array_equal(np.ones((nx,nx)),mask) == True:

        inpainted_image = image

    else:
        mask = np.asarray(mask, dtype=image.dtype) # Matching types needed for jit
        masked_image = image*mask
        zeros = np.where(mask == 0.)
        x_0 = zeros[0]
        y_0 = zeros[1]
        i = 0
        inpainted_image = masked_image

        for j in range(0,n_inpaint,1):

            for i in range(0,len(x_0),1):

                x = x_0[i]
                y = y_0[i]
                c = 0
                value = 0.

                if nx > x-1 >= 0 and nx > y-1 >= 0:

                    value += masked_image[x-1,y-1]
                    c += 1

                if nx > y-1 >= 0:

                    value += masked_image[x,y-1]
                    c += 1

                if nx > x+1 >= 0 and nx > y-1 >= 0:

                    value += masked_image[x+1,y-1]
                    c += 1

                if nx > x-1 >= 0:

                    value += masked_image[x-1,y]
                    c += 1

                if nx > x+1 >= 0:

                    value += masked_image[x+1,y]
                    c += 1

                if nx > x-1 >= 0 and nx > y+1 >= 0:

                    value += masked_image[x-1,y+1]
                    c += 1

                if nx > y+1 >= 0:

                    value += masked_image[x,y+1]
                    c += 1

                if nx > x+1 >= 0 and nx > y+1 >= 0:

                    value += masked_image[x+1,y+1]
                    c += 1

                inpainted_image[x,y] = value/c

    return inpainted_image

#Expand CAR map by 1 degree (physical distance, not coordinate difference) from all four edges:

def get_expanded_map_car(full_map, radec, expansion_deg=1.0):

    from pixell import utils as pixell_utils

    [ra_min,ra_max,dec_min,dec_max] = radec

    ra_min_expanded = ra_min - expansion_deg
    ra_max_expanded = ra_max + expansion_deg

    def move_point_constant_dec(dec,ra,expansion_degree):

        # Convert to radians
        theta = np.radians(90 - dec)  # colatitude
        phi = np.radians(ra)

        # Convert to 3D vector
        vec = hp.ang2vec(theta, phi)

        # Define rotation: about Z axis by -1 deg (westward), keeping Dec constant
        rot = hp.Rotator(rot=[expansion_degree/np.sin(theta), 0, 0], deg=True)
        vec_rot = rot(vec)

        # Convert back to angles
        theta_new, phi_new = hp.vec2ang(vec_rot)
        ra_new = np.degrees(phi_new) % 360  # wraparound

        return ra_new

    def ra_rightmost(ra1, ra2):
        # Returns the RA value (in degrees) that is most to the right/east
        diff = (ra2 - ra1) % 360
        return ra2 if diff < 180 else ra1

    def ra_leftmost(ra1, ra2):
        # Returns the RA value (in degrees) that is most to the left/west
        diff = (ra2 - ra1) % 360
        return ra1 if diff < 180 else ra2

    ra_topleft_expanded = move_point_constant_dec(dec_max,ra_min,expansion_degree=expansion_deg)
    ra_topright_expanded = move_point_constant_dec(dec_max,ra_max,expansion_degree=-expansion_deg)
    ra_bottomleft_expanded = move_point_constant_dec(dec_min,ra_min,expansion_degree=expansion_deg)
    ra_bottomright_expanded = move_point_constant_dec(dec_min,ra_max,expansion_degree=-expansion_deg)

    # print(ra_min,ra_max)
    # print(ra_topleft_expanded,ra_bottomleft_expanded)
    # print(ra_topright_expanded,ra_bottomright_expanded)

    ra_min_expanded = ra_leftmost(ra_bottomleft_expanded,ra_topleft_expanded)[0]
    ra_max_expanded = ra_rightmost(ra_topright_expanded,ra_bottomright_expanded)[0]

    dec_min_expanded = dec_min - expansion_deg
    dec_max_expanded = dec_max + expansion_deg

    # print(dec_min,dec_min_expanded)
    # print(dec_max,dec_max_expanded)
    # print(ra_min,ra_min_expanded)
    # print(ra_max,ra_max_expanded)

    submap_expanded = full_map.submap([[dec_min_expanded*pixell_utils.degree,ra_min_expanded*pixell_utils.degree], [dec_max_expanded*pixell_utils.degree,ra_max_expanded*pixell_utils.degree]])

    dec_map, ra_map = submap_expanded.posmap()
    dec_map = np.degrees(dec_map)
    ra_map = np.degrees(ra_map)


    mask = ((dec_map >= dec_min) & (dec_map <= dec_max) &
            ((ra_map - ra_min) % (360.) <= (ra_max - ra_min) % (360.))).astype(float)

    return submap_expanded, mask