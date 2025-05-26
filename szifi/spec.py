import numpy as np
import os
from scipy import interpolate
import pymaster as nmt
from szifi import maps
import scipy.ndimage as ndimage


class power_spectrum:

    def __init__(self, pix, mask=None, cm_compute=False, cm_save=False,
    cm_name=None, bins=None, bin_fac=4., new_shape=None, cm_compute_scratch=False):

        if new_shape is not None:
            pix = maps.degrade_pix(pix, new_shape)
            mask = maps.degrade_map(mask, new_shape)

        if mask is None:

            mask = np.ones((pix.nx,pix.ny))

        if bins is None:

            l0_bins = np.arange(pix.nx/bin_fac)*bin_fac*np.pi/(pix.nx*pix.dx)
            lf_bins = (np.arange(pix.nx/bin_fac)+1)*bin_fac*np.pi/(pix.nx*pix.dx)
            bins = nmt.NmtBinFlat(l0_bins,lf_bins)

        self.l0_bins = l0_bins
        self.lf_bins = lf_bins
        self.pix = pix
        self.mask = mask
        self.bins = bins
        self.new_shape = new_shape
        self.n_modes_per_bin = np.zeros(len(self.l0_bins))

        self.ell_eff = self.bins.get_effective_ells()
        self.w00 = None

        if cm_compute == True:

            if cm_name == None:

                print("Coupling matrix not found")
                self.get_coupling_matrix()

                if cm_save == True:

                    self.w00.write_to(cm_name)

            elif os.path.isfile(cm_name) == False:

                print("Coupling matrix not found")
                self.get_coupling_matrix()

                if cm_save == True:

                    self.w00.write_to(cm_name)

            elif os.path.isfile(cm_name) == True:

                print("Coupling matrix found")

                if cm_compute_scratch == True:

                    os.remove(cm_name)
                    print("Coupling matrix recomputation")
                    self.get_coupling_matrix()

                    if cm_save == True:

                        self.w00.write_to(cm_name)

                else:

                    self.w00 = nmt.NmtWorkspaceFlat()
                    self.w00.read_from(cm_name)


    def get_coupling_matrix(self):

        f0 = nmt.NmtFieldFlat(self.pix.nx*self.pix.dx,self.pix.ny*self.pix.dy,self.mask,[np.zeros((self.pix.nx,self.pix.ny))])
        w00 = nmt.NmtWorkspaceFlat()
        w00.compute_coupling_matrix(f0,f0,self.bins)
        self.w00 = w00

    #Note: map1 is UNMASKED

    def get_power_spectrum(self,map1,map2=None,decouple_type="none",implementation="pymaster"):
        if self.new_shape is not None:
            map1 = maps.degrade_map(map1, self.new_shape)
            map2 = maps.degrade_map(map2, self.new_shape)
        if map2 is None:

            map2 = map1

        if implementation == "pymaster":

            f1 = nmt.NmtFieldFlat(self.pix.nx*self.pix.dx,self.pix.ny*self.pix.dy,self.mask,[map1])
            f2 = nmt.NmtFieldFlat(self.pix.nx*self.pix.dx,self.pix.ny*self.pix.dy,self.mask,[map2])
            cl00_coupled = nmt.compute_coupled_cell_flat(f1,f2,self.bins)

            if decouple_type == "none":

                ret = (self.ell_eff,cl00_coupled[0])

            elif decouple_type == "master":

                if self.w00 is None:

                    self.get_coupling_matrix()

                ret = (self.ell_eff,self.w00.decouple_cell(cl00_coupled)[0])

            elif decouple_type == "fsky":

                fsky = np.sum(self.mask)/(self.pix.nx*self.pix.ny)
                m1 = np.sum(self.mask**2)/(self.pix.nx*self.pix.ny)
                m2 = np.sum(self.mask**4)/(self.pix.nx*self.pix.ny)
                fsky = fsky*m1**2/m2

                ret = (self.ell_eff,cl00_coupled[0]/fsky)

        elif implementation == "custom":

            map1_fft = maps.get_fft(map1,self.pix)
            map2_fft = maps.get_fft(map2,self.pix)
            spec_map = np.conjugate(map1_fft)*map2_fft
            ell_map = ell_map = maps.rmap(self.pix).get_ell()
            (ell_vec_eff,spec_tensor_binned,n_modes_per_bin,n_modes_per_bin_map) = get_binned_spec(spec_map,ell_map,self.l0_bins,self.lf_bins)
            self.ell_eff = ell_vec_eff
            self.n_modes_per_bin = n_modes_per_bin
            self.n_modes_per_bin_map =  n_modes_per_bin_map
            self.ell_map = ell_map

            ret = (ell_vec_eff,spec_tensor_binned)

        return ret

def get_binned_spec(spec_map,ell_map,bins_low,bins_high):

    spec_tensor_binned = np.zeros(len(bins_low))
    n_modes_per_bin = np.zeros(len(bins_low))
    n_modes_per_bin_map = np.zeros(ell_map.shape[0]*ell_map.shape[1])
    spec_map_shape = spec_map.shape

    for i in range(0,len(bins_low)):

        ell_map = ell_map.flatten()
        indices = np.where((ell_map < bins_high[i]) & (ell_map >= bins_low[i]))[0]


        spec_tensor_binned[i] = np.mean(spec_map.flatten()[indices])
        n_modes_per_bin[i] = len(indices)
        n_modes_per_bin_map[indices] = len(indices)

    n_modes_per_bin_map = n_modes_per_bin_map.reshape(spec_map_shape)
    spec_map = spec_map.reshape(spec_map_shape)

    ell_vec_eff = 0.5*(bins_low+bins_high)

    return (ell_vec_eff,spec_tensor_binned,n_modes_per_bin,n_modes_per_bin_map)


class cross_spec:

    def __init__(self,freqs):

        self.freqs = freqs
        self.n_freqs = len(self.freqs)
        self.spec_tensor = None
        self.ell_vec = None

    def get_cross_spec(self,pix,t_map=None,t_map_2=None,mask=None,bin_fac=4., new_shape=None, ps=None,
    decouple_type="master",inpaint_flag=False,mask_point=None,
    lsep=3000,implementation="pymaster"): #estimates cross spectra. exp only if theory == True

        pix = maps.degrade_pix(pix, new_shape)
        # Other degrading in power_spectrum rather than here

        if mask is None:

            mask = np.ones((pix.nx,pix.ny))

        if mask_point is None:

            mask_point = np.ones((pix.nx,pix.ny))

        if t_map_2 is None:

            t_map_2 = t_map

        if ps is None:

            ps = power_spectrum(pix, mask=mask, cm_compute=False, bin_fac=bin_fac, new_shape=new_shape)

        ell = ps.ell_eff[1:]

        n_freq = self.n_freqs
        spec_tensor = np.zeros((len(ell),n_freq,n_freq))
        self.ps = ps

        if ps.new_shape != new_shape:

            raise ValueError(f"cross_spec new_shape={new_shape} and ps {ps.new_shape} should be equal")

        for i in range(0,n_freq):

            for j in range(0,n_freq):

                if j >= i:

                    ell,cl = ps.get_power_spectrum(t_map[:,:,i],
                    map2=t_map_2[:,:,j],decouple_type=decouple_type,
                    implementation=implementation)
                    if inpaint_flag == True:

                        ell,cl = correct_for_inpainting((ell,cl),mask_point,lsep=lsep)

                    ell = ell[1:]
                    cl = cl[1:]
                    spec_tensor[:,i,j] = cl

                else:

                    spec_tensor[:,i,j] = spec_tensor[:,j,i]

        self.spec_tensor = spec_tensor
        self.ell_vec = ell

        return ell,spec_tensor # len(ell) x n_freq x n_freq tensor


    def get_cov(self,pix,t_map=None,mask=None,bin_fac=4,new_shape=None,ps=None,
    decouple_type="master",interp_type="nearest",cov_type="isotropic"):
        
        if cov_type == "isotropic":

            if self.spec_tensor is None:

                ell_vec,spec_tensor = self.get_cross_spec(pix,t_map=t_map,
                                                          mask=mask,
                                                          bin_fac=bin_fac,
                                                          new_shape=new_shape,
                                                          ps=ps,
                                                          decouple_type=decouple_type)

            else:

                ell_vec = self.ell_vec
                spec_tensor = self.spec_tensor

            pix = maps.degrade_pix(pix, new_shape)
            ell_map = maps.rmap(pix).get_ell()
            cov_tensor = interp_cov(ell_map,ell_vec,spec_tensor,interp_type=interp_type)

            # import pylab as pl
            # pl.figure()
            # pl.imshow(cov_tensor[:,:,0,0].real)
            # pl.savefig("/home/iz221/szifi/test_files/figures/cov_isotropic.pdf")
            # pl.show()
            # quit()

        else:

            cov_tensor = get_cov_anisotropic(t_map,pix,mask=mask,cov_type=cov_type)

        return cov_tensor # nx x ny x n_freq x n_freq covariance tensor

    def get_inv_cov(self,pix,t_map=None,mask=None,bin_fac=4,new_shape=None,ps=None,
                    decouple_type="master",interp_type="nearest",cov_type="isotropic"):

        return np.linalg.inv(self.get_cov(pix,t_map=t_map,mask=mask,bin_fac=bin_fac,new_shape=new_shape,
        ps=ps,decouple_type=decouple_type,interp_type=interp_type,
        cov_type=cov_type))

def get_cov_anisotropic(t_map,pix,mask=None,cov_type=None):

    n_freq = t_map.shape[2]
    cov_tensor = np.zeros((pix.nx,pix.ny,n_freq,n_freq),dtype=complex)

    for i in range(0,n_freq):

        for j in range(0,n_freq):

            if j >= i:

                cov_tensor[:,:,i,j] = get_anisotropic_ps(t_map[:,:,i],t_map[:,:,j],pix,type=cov_type,mask=mask)

            else:

                cov_tensor[:,:,i,j] = cov_tensor[:,:,j,i]

    return cov_tensor


def get_anisotropic_ps(map1,map2,pix,type="boxcar",mask=None):
                
        power_spectrum = np.real(np.conjugate(maps.get_fft(map1*mask,pix))*maps.get_fft(map2*mask,pix))

        if type == "anisotropic_boxcar":

            nx_box = 40
            ny_box = 40

            kernel = np.ones((nx_box,ny_box))/(nx_box*ny_box)
            power_spectrum = np.fft.ifftshift(ndimage.convolve(np.fft.fftshift(power_spectrum),kernel,mode='reflect')).real

        elif type == "anisotropic_gaussian":

            gaussian_sigma = np.array([5,5])
            power_spectrum = np.fft.ifftshift(ndimage.gaussian_filter(np.fft.fftshift(power_spectrum),sigma=gaussian_sigma,mode="reflect")).real

        else:

            print("Power spectrum estimation type not supported")
            power_spectrum = None

        return power_spectrum



def interp_cov(ell_interp,ell,cov_tensor,interp_type="nearest"):

    (d1,d2,d3) = cov_tensor.shape
    (nx,ny) = ell_interp.shape
    interp_tensor = np.zeros((nx,ny,d2,d3))

    for i in range(0,d2):

        for j in range(0,d3):

            interp_tensor[:,:,i,j] = interpolate.interp1d(ell,cov_tensor[:,i,j],kind=interp_type,bounds_error=False,fill_value="extrapolate")(ell_interp)

    return interp_tensor


def get_camb_cltt(exp=None,freq=0,freq2=None,beams="gaussian"):

    ell,cl_tt = np.load("camb_cl_tt.npy",allow_pickle=True)
    cl_tt = cl_tt*2.*np.pi/(ell*(ell+1.))
    cl_tt[0] = 0.

    if exp is not None:

        if freq2 is None:

            freq2 = freq

        FWHM = exp.FWHM

        if beams == "gaussian":

            cl_tt *= maps.get_bl(FWHM[freq],ell)*maps.get_bl(FWHM[freq2],ell)

        elif beams == "real":

            beam_i = exp.get_beam(freq)[1]
            beam_j = exp.get_beam(freq2)[1]
            ell_beam = exp.get_beam(freq)[0]
            cl_tt *= np.interp(ell,ell_beam,beam_i)*np.interp(ell,ell_beam,beam_j)

    return ell,cl_tt

def bin_spec(ell,cl,bins_low,bins_high):

    cl_binned = np.zeros(len(bins_low))

    for i in range(0,len(bins_low)):

        indices = np.where((ell < bins_high[i]) & (ell >= bins_low[i]))[0]
        cl_binned[i] = np.mean(cl[indices])

    ell_eff = 0.5*(bins_low+bins_high)

    return ell_eff,cl_binned


def correct_for_inpainting(input_spectrum,mask_point,lsep=3000):

    [ell,cl] = input_spectrum
    fsky = np.sum(mask_point)/(mask_point.shape[0]*mask_point.shape[1])
    cl_corrected = np.zeros(len(ell))
    indices_low = np.where(ell>lsep)[0]
    indices_high = np.where(ell<=lsep)[0]
    cl_corrected[indices_low] = cl[indices_low]/fsky
    cl_corrected[indices_high] = cl[indices_high]

    return (ell,cl_corrected)

class ps_flat_sky:

    def __init__(self,pix,mask,ell_bins_edges=None,fac=4):

        self.pix = pix
        self.mask = mask
        Lx = pix.nx*pix.dx
        Ly = pix.ny*pix.dy

        if ell_bins_edges == None:

            l0_bins = np.arange(pix.nx/fac)*fac*np.pi/(pix.nx*pix.dx)
            lf_bins = (np.arange(pix.nx/fac)+1)*fac*np.pi/(pix.nx*pix.dx)
            bins = nmt.NmtBinFlat(l0_bins, lf_bins)
            ells_uncoupled = bins.get_effective_ells()
            self.bins = bins

        f0 = nmt.NmtFieldFlat(Lx,Ly,mask,[np.zeros((self.pix.nx,self.pix.ny))])
        w00 = nmt.NmtWorkspaceFlat()
        w00.compute_coupling_matrix(f0,f0,self.bins)
        self.w00 = w00
        self.ells_uncoupled = ells_uncoupled

    def get_ps(self,map,decouple=True):

        f1 = nmt.NmtFieldFlat(self.pix.nx*self.pix.dx,self.pix.ny*self.pix.dy,self.mask,[map])
        cl00_coupled = nmt.compute_coupled_cell_flat(f1,f1,self.bins)

        if decouple == True:

            cl_uncoupled = self.w00.decouple_cell(cl00_coupled)[0]
            
        return self.ells_uncoupled,cl_uncoupled
