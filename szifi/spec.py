import numpy as np
import os
from scipy import interpolate
import pymaster as nmt
from szifi import maps

class power_spectrum:

    def __init__(self, pix, mask=None, cm_compute=False, cm_save=False,
    cm_name=None, bins=None, bin_fac=4., new_shape=None, cm_compute_scratch=False):

        if new_shape is not None:
            pix = maps.degrade_pix(pix, new_shape)
            mask = maps.degrade_map(mask, new_shape)

        if mask is None:

            mask = np.ones((pix.nx,pix.ny))

        if bins is None:

            if bin_fac == 1.: #actually 1

                ell = np.sort(np.unique(maps.rmap(pix).get_ell().flatten()))
                l0_bins = ell - 1.
                lf_bins = ell + 1.

            else:

                l0_bins = np.arange(pix.nx/bin_fac)*bin_fac*np.pi/(pix.nx*pix.dx)
                lf_bins = (np.arange(pix.nx/bin_fac)+1)*bin_fac*np.pi/(pix.nx*pix.dx)

                #l0_bins = np.linspace(50,10000,256)
                #lf_bins = np.append(l0_bins[1:],10100)

                #print(len(l0_bins))


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

    def __init__(self,freqs,spec_matrix=None,ell_vec=None):

        self.freqs = freqs
        self.n_freqs = len(self.freqs)
        self.spec_tensor = None
        self.ell_vec = None

    def get_cross_spec(self,pix,t_map=None,t_map_2=None,mask=None,bin_fac=4., new_shape=None, ps=None,theory=False,
    lmax=30000,decouple_type="master",inpaint_flag=False,mask_point=None,
    lsep=3000,beam="gaussian",implementation="pymaster",exp=None,cib_flag=False,
    noise_flag=True,cmb_flag=True,tsz_flag=False,tsz_cib_flag=False): #estimates cross spectra. exp only if theory == True

        pix = maps.degrade_pix(pix, new_shape)
        # Other degrading in power_spectrum rather than here

        if mask is None:

            mask = np.ones((pix.nx,pix.ny))

        if mask_point is None:

            mask_point = np.ones((pix.nx,pix.ny))

        if t_map_2 is None:

            t_map_2 = t_map

        if theory == True:

            ell = np.arange(2,11000)[100:]

        elif theory == False:

            if ps is None:

                ps = power_spectrum(pix, mask=mask, cm_compute=False, bin_fac=bin_fac, new_shape=new_shape)

            ell = ps.ell_eff[1:]

        n_freq = self.n_freqs
        spec_tensor = np.zeros((len(ell),n_freq,n_freq))
        self.ps = ps
        if ps.new_shape != new_shape:
            raise ValueError(f"cross_spec new_shape={new_shape} and ps {ps.new_shape} should be equal")
        if theory == False:

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

        if theory == True:

            spec_tensor = cross_spec_theory(ell,exp,freqs=self.freqs,cmb_flag=cmb_flag,
            cib_flag=cib_flag,tsz_flag=tsz_flag,noise_flag=False,tsz_cib_flag=tsz_cib_flag).cross_spec_tensor
            spec_tensor = spec_tensor*get_beam_tensor(ell,exp,self.freqs,beam_type=beam)
            spec_tensor = spec_tensor + cross_spec_noise(ell,exp,freqs=self.freqs).cross_spec_tensor


        self.spec_tensor = spec_tensor
        self.ell_vec = ell

        return ell,spec_tensor # len(ell) x n_freq x n_freq tensor


    def get_cov(self,pix,t_map=None,mask=None,bin_fac=4,new_shape=None,ps=None,theory=False,cmb_flag=True,
    lmax=30000,decouple_type="master",interp_type="nearest",beam="gaussian",exp=None,
    cib_flag=False,tsz_flag=False,noise_flag=True,tsz_cib_flag=False):

        if self.spec_tensor is None:

            ell_vec,spec_tensor = self.get_cross_spec(pix,t_map=t_map,mask=mask,bin_fac=bin_fac, new_shape=new_shape,
            ps=ps,theory=theory,cmb_flag=cmb_flag,lmax=lmax,decouple_type=decouple_type,beam=beam,exp=exp,
            cib_flag=cib_flag,noise_flag=noise_flag,tsz_flag=tsz_flag,tsz_cib_flag=tsz_cib_flag)

        else:

            ell_vec = self.ell_vec
            spec_tensor = self.spec_tensor

        pix = maps.degrade_pix(pix, new_shape)
        ell_map = maps.rmap(pix).get_ell()
        cov_tensor = interp_cov(ell_map,ell_vec,spec_tensor,interp_type=interp_type)

        return cov_tensor # nx x ny x n_freq x n_freq covariance tensor

    def get_inv_cov(self,pix,t_map=None,mask=None,bin_fac=4,new_shape=None,ps=None,theory=False,
    cmb_flag=True,lmax=30000,decouple_type="master",interp_type="nearest",beam="gaussian",exp=None,
    cib_flag=False,tsz_flag=False,noise_flag=True,tsz_cib_flag=False):

        return np.linalg.inv(self.get_cov(pix,t_map=t_map,mask=mask,bin_fac=bin_fac,new_shape=new_shape,
        ps=ps,theory=theory,cmb_flag=cmb_flag,lmax=lmax,decouple_type=decouple_type,interp_type=interp_type,
        beam=beam,exp=exp,cib_flag=cib_flag,noise_flag=noise_flag,tsz_flag=tsz_flag,tsz_cib_flag=tsz_cib_flag))

    def get_inv_cov_anisotropic(self,t_map,pix,mask=None):

        return np.linalg.inv(self.get_cov_anisotropic(t_map,pix,mask=mask))

    def get_cov_anisotropic(self,t_map,pix,mask=None):

        if mask == None:

            mask = np.ones((pix.nx,pix.ny))

        n_freq = t_map.shape[2]
        cov_tensor = np.zeros((pix.nx,pix.ny,n_freq,n_freq),dtype=complex)

        for i in range(0,n_freq):

            for j in range(0,n_freq):

                if j >= i:

                    cov_tensor[:,:,i,j] = np.conjugate(maps.get_fft(t_map[:,:,i],pix))*maps.get_fft(t_map[:,:,j],pix)

                else:

                    cov_tensor[:,:,i,j] = cov_tensor[:,:,j,i]

        return cov_tensor

def interp_cov(ell_interp,ell,cov_tensor,interp_type="nearest"):

    (d1,d2,d3) = cov_tensor.shape
    (nx,ny) = ell_interp.shape
    interp_tensor = np.zeros((nx,ny,d2,d3))

    for i in range(0,d2):

        for j in range(0,d3):

            interp_tensor[:,:,i,j] = interpolate.interp1d(ell,cov_tensor[:,i,j],kind=interp_type,bounds_error=False,fill_value="extrapolate")(ell_interp)
            #interp_tensor[:,:,i,j] = np.interp(ell_interp,ell,cov_tensor[:,i,j])

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

#Returns theory cross spectra for a given experiment. NOT convolved. Output units in muK

class cross_spec_theory:

    def __init__(self,ell,exp,freqs=None,cmb_flag=True,cib_flag=False,tsz_flag=False,
    noise_flag=True,tsz_cib_flag=False):

        if freqs is None:

            freqs = np.arange(len(exp.nu_eff))

        self.ell = ell
        self.freqs = freqs
        self.exp = exp
        self.cross_spec_tensor = np.zeros((len(self.ell),len(self.freqs),len(self.freqs)))
        #self.class_sz_file_name = "class_sz_tsz_cib_planck.npy"
        self.class_sz_file_name = "class_sz_tsz_cib_all_exps.npy"

        if cmb_flag == True:

            ell_cmb,cl_cmb = get_camb_cltt(exp=None)

            for i in range(0,len(self.freqs)):

                for j in range(0,len(self.freqs)):

                    self.cross_spec_tensor[:,i,j] = self.cross_spec_tensor[:,i,j] + np.interp(self.ell,ell_cmb,cl_cmb)

        if tsz_flag == True:

            [cl_sz,cl_cib_cib,cl_tsz_cib] = np.load(self.class_sz_file_name,allow_pickle=True)

            ell_tsz = np.asarray(cl_sz["ell"])
            cl_tsz = np.asarray(cl_sz["1h"]) + np.asarray(cl_sz["2h"])

            for i in range(0,len(self.freqs)):

                for j in range(0,len(self.freqs)):

                    self.cross_spec_tensor[:,i,j] = self.cross_spec_tensor[:,i,j] + np.interp(self.ell,ell_tsz,cl_tsz)*self.exp.tsz_f_nu[self.freqs[i]]*self.exp.tsz_f_nu[self.freqs[j]]

        if cib_flag == True:

            [cl_sz,cl_cib_cib,cl_tsz_cib] = np.load(self.class_sz_file_name,allow_pickle=True)

            cib_tensor = np.zeros(self.cross_spec_tensor.shape)

            for i in range(0,len(self.freqs)):

                for j in range(0,len(self.freqs)):

                    freq_name_i = int(np.floor(exp.nu_eff[self.freqs[i]]/1e9))
                    freq_name_j = int(np.floor(exp.nu_eff[self.freqs[j]]/1e9))

                    freq_name_a = np.max([freq_name_i,freq_name_j])
                    freq_name_b = np.min([freq_name_i,freq_name_j])

                    cross_name = str(freq_name_a) + "x" + str(freq_name_b)

                    ell_cib = np.asarray(cl_cib_cib[cross_name]["ell"])
                    cl_cib = np.asarray(cl_cib_cib[cross_name]["2h"]) + np.asarray(cl_cib_cib[cross_name]["1h"])
                    factor = ell_cib*(ell_cib+1.)/(2.*np.pi)
                    cl_cib = cl_cib/factor

                    cl_cib_muK = np.interp(self.ell,ell_cib,cl_cib)*1e-12*MJysr_to_muK_factor(self.exp.nu_eff[self.freqs[i]])*MJysr_to_muK_factor(self.exp.nu_eff[self.freqs[j]])

                    cib_tensor[:,i,j] = cib_tensor[:,i,j] + cl_cib_muK

            self.cross_spec_tensor = self.cross_spec_tensor + cib_tensor


        if tsz_cib_flag == True:

            [cl_sz,cl_cib_cib,cl_tsz_cib] = np.load(self.class_sz_file_name,allow_pickle=True)

            cross_matrix = np.zeros(self.cross_spec_tensor.shape)

            for i in range(0,len(self.freqs)):

                for j in range(0,len(self.freqs)):

                    freq_name = str(int(np.floor(exp.nu_eff[self.freqs[i]]/1e9)))

                    ell_cib = np.asarray(cl_tsz_cib[freq_name]["ell"])
                    cl_cib = np.asarray(cl_tsz_cib[freq_name]["2h"]) #+ np.asarray(cl_tsz_cib[freq_name]["1h"])
                    factor = ell_cib*(ell_cib+1.)/(2.*np.pi)
                    cl_cib = cl_cib/factor

                    cl_tsz_cib_muK = np.interp(self.ell,ell_cib,cl_cib)*1e-12*MJysr_to_muK_factor(self.exp.nu_eff[self.freqs[i]])*self.exp.tsz_f_nu[self.freqs[j]]

                    cross_matrix[:,i,j] = cl_tsz_cib_muK

            cross_matrix = cross_matrix + cross_matrix.swapaxes(1,2)

            self.cross_spec_tensor = self.cross_spec_tensor + cross_matrix

        if noise_flag == True:

            for i in range(0,len(self.freqs)):

                self.cross_spec_tensor[:,i,i] = self.cross_spec_tensor[:,i,i] + maps.get_nl(self.exp.noise_levels[self.freqs[i]],0.,self.ell)

class cross_spec_noise:

    def __init__(self,ell,exp,freqs=None):

        if freqs is None:

            freqs = np.arange(len(exp.nu_eff))

        self.ell = ell
        self.freqs = freqs
        self.exp = exp
        self.cross_spec_tensor = np.zeros((len(self.ell),len(self.freqs),len(self.freqs)))

        for i in range(0,len(self.freqs)):

            self.cross_spec_tensor[:,i,i] = self.cross_spec_tensor[:,i,i] + maps.get_nl(self.exp.noise_levels[self.freqs[i]],0.,self.ell)


def get_beam_tensor(ell,exp,freqs,beam_type="gaussian"):

    beam_tensor = np.zeros((len(ell),len(freqs),len(freqs)))

    for i in range(0,len(freqs)):

        for j in range(0,len(freqs)):

             if beam_type == "gaussian":

                 beam_tensor[:,i,j] = maps.get_bl(exp.FWHM[freqs[i]],ell)*maps.get_bl(exp.FWHM[freqs[j]],ell)

             elif beam_type == "real":

                 ell_i,beam_i = exp.get_beam(freqs[i])
                 ell_j,beam_j = exp.get_beam(freqs[j])
                 beam_tensor[:,i,j] = np.interp(ell,ell_i,beam_i)*np.interp(ell,ell_j,beam_j)

    return beam_tensor
