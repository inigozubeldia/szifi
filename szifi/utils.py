import numpy as np
from numba import jit
from szifi import maps, mmf
import scipy.signal as sg

#Various utility functions

#Calculate covariance (or ILC) bias on signal-to-noise

def get_q_bias(pix,tem,signal,cov_mf_true,cov_noi_true,n_modes_per_bin_map,type="q"):

    tem_fft = maps.get_fft(tem,pix)
    signal_fft = maps.get_fft(signal,pix)

    n0 = filter_sum_1(tem_fft,tem_fft,cov_mf_true)

    map1 = (signal_fft+np.conjugate(signal_fft))*cov_noi_true*np.conjugate(tem_fft)
    map2 = (signal_fft+np.conjugate(signal_fft))*cov_noi_true*np.conjugate(tem_fft)*np.abs(tem_fft)**2
    map1 = div0(map1,cov_mf_true**2*n_modes_per_bin_map)
    map2 = div0(map2,cov_mf_true**3*n_modes_per_bin_map)

    sum1 = np.sum(map1).real
    sum2 = np.sum(map2).real

    if type == "q":

        bias1 = -sum1/n0**0.5
        bias2 = 0.5*sum2/n0**1.5

    elif type == "y0":

        bias1 = -sum1/n0
        bias2 = sum2/n0**2

    return bias1,bias2

#Calculate covariance (or ILC) bias on signal-to-noise variance

def get_q_std_bias(pix,tem,signal,cov_mf_true,cov_noi_true):

    tem_fft = maps.get_fft(tem,pix)
    signal_fft = maps.get_fft(signal,pix)
    n0 = filter_sum_1(tem_fft,tem_fft,cov_mf_true)
    n_bias = filter_sum_1(tem_fft,tem_fft*cov_noi_true,(cov_mf_true)**2)
    n_bias = filter_sum_1(tem_fft,tem_fft*cov_noi_true,(cov_mf_true)**2)
    sigma_unbiased = 1./np.sqrt(n0)
    sigma_biased = 1./np.sqrt(n_bias)

    return sigma_unbiased/sigma_biased

#Returns MMF y0 at centre (just one cluster)

def get_mmf_y0(tmap,tem,inv_cov,pix,nfreq=6,ell_filter=None,mmf_type="standard",
cmmf_prec=None):

    if ell_filter == None:

        ell_filter = [0.,1./pix.dx*10.]

    n_freqs = tmap.shape[2]
    y_map = np.zeros((pix.nx,pix.ny))
    norm_map = np.zeros((pix.nx,pix.ny))

    tem_fft = maps.get_fft_f(tem,pix)
    tem_fft = tem_fft

    tem = maps.get_ifft_f(maps.filter_fft_f(tem_fft,pix,ell_filter),pix).real

    y0_vec = np.zeros(nfreq+1)
    std_vec = np.zeros(nfreq+1)

    filter_fft = mmf.get_tem_conv_fft(pix,tem_fft,inv_cov,mmf_type=mmf_type,cmmf_prec=cmmf_prec)
    filter_fft = maps.filter_fft_f(filter_fft,pix,ell_filter)

    for i in range(nfreq):

        y = sg.fftconvolve(tmap[:,:,i],filter_fft[:,:,i],mode='same')*pix.dx*pix.dy
        sigma2 = sg.fftconvolve(tem[:,:,i],filter_fft[:,:,i],mode='same')*pix.dx*pix.dy

        y_map += y
        norm_map += sigma2

        y0_vec[i] = y[pix.nx//2,pix.ny//2]/np.max(sigma2)
        std_vec[i] = 1/np.sqrt(np.max(sigma2))

    norm = np.max(norm_map)

    y0_vec[-1] = y_map[pix.nx//2,pix.ny//2]/np.max(norm_map)
    std_vec[-1] = 1/np.sqrt(np.max(norm_map))

    if mmf_type == "spectrally_constrained":

        std_vec[-1] = get_cmmf_std(pix,mmf_type,inv_cov,filter_fft,np.arange(n_freqs),norm)

    return y0_vec,std_vec

def get_mf_map(obs,tem,noi,pix):

    tem_fft = maps.get_fft(tem,pix)
    tem_conv = maps.get_ifft(tem_fft/noi,pix).real

    map_convolution = sg.fftconvolve(obs,tem_conv,mode='same')*pix.dx*pix.dy
    norm = np.max(sg.fftconvolve(tem,tem_conv,mode='same'))*pix.dx*pix.dy
    est_map = map_convolution/norm
    std = 1./np.sqrt(norm)

    return est_map,std

def filter_sum(rec,tem,noi):

    map1 = np.conjugate(rec)*tem
    map1 = div0(map1,noi)
    map2 = np.conjugate(tem)*tem
    map2 = div0(map2,noi)

    sum_1 = np.sum(map1).real
    sum_2 = np.sum(map2).real

    normalisation_estimate = sum_1/sum_2
    normalisation_variance = 1./sum_2

    return (normalisation_estimate,normalisation_variance)

def div0(a,b):

    with np.errstate(divide='ignore',invalid='ignore'):

        c = np.true_divide(a,b)
        c[~np.isfinite(c)] = 0

    return c

def get_cmmf_std(pix,mmf_type,inv_cov,filter_fft,norm):

    freqs = np.arange(filter_fft.shape[2])

    cov = invert_cov(inv_cov)

    filter_fft_2 = get_inv_cov_conjugate(filter_fft,cov)

    filter_2 = maps.get_ifft_f(filter_fft_2,pix).real
    filter = maps.get_ifft_f(filter_fft,pix).real

    var_map = np.zeros((pix.nx,pix.ny),dtype=complex)

    for i in range(len(freqs)):

        var_map += sg.fftconvolve(filter[:,:,freqs[i]],filter_2[:,:,i],mode='same')*pix.dx*pix.dy

    var_map = var_map/norm**2
    var = var_map[pix.nx//2,pix.nx//2].real
    std = np.sqrt(var)

    return std

def invert_cov(cov):

    inv_cov = np.zeros(cov.shape)

    for i in range(0,cov.shape[0]):

        for j in range(0,cov.shape[1]):

            if not np.any(cov[i,j,:,:]) == False:

                if cov[i,j,:,:].any() == 0:

                    inv_cov[i,j,:,:] = cov[i,j,:,:]

            #    if np.linalg.det(inv_cov[i,j,:,:]) == 0. and len(inv_cov[i,j,:,:]) > 1 and np.sum(inv_cov[i,j,:,:]) > 0.:

                #    print(inv_cov[i,j,:,:])

                else:

                    inv_cov[i,j,:,:] = np.linalg.inv(cov[i,j,:,:])

    #inv_cov =  np.linalg.inv(cov)

    return inv_cov

def get_inv_cov_conjugate(tem_fft,inv_cov):

    return np.einsum('dhi,dhij->dhj',tem_fft,inv_cov)

def get_inv_cov_dot(a,inv_cov,b):

    conjugate = get_inv_cov_conjugate(a,inv_cov)

    return np.einsum('dhi,dhi->dh',conjugate,b)

def gaussian_1d(x,mu,sigma):

    y = (x-mu)/sigma

    return np.exp(-0.5*y**2)/(np.sqrt(2.*np.pi*sigma**2))
