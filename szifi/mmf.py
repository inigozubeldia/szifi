import numpy as np
import pylab as pl
import scipy.signal as sg
from astropy.cosmology import Planck15
from sklearn.cluster import DBSCAN
import time
import maps
import spec
import cat
import model
import inpaint
import sphere
import healpy as hp
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import time
from numba import jit
import warnings
warnings.filterwarnings("ignore")
from sys import getsizeof
import expt

#theta_range defined w.r.t. to Cartesian coordinates, no matrix coordinates.
#mask is nx x ny array (not time freq)

#mask_map : mask used to compute power spectra and to apply on temperature maps before filtering.
#mask_select: masked applied on q map to detect clusters.

class mmf_detection:

    def __init__(self,pix,theta_range=None,q_th=4.5,
    concentration=None,refinement=True,
    it=True,cosmology=Planck15,q_th_noise=6.,decouple=True,
    estimate_spec="estimate",fac=4.,save_mask=False,
    cm_compute_scratch=False,decouple_type="master",mask_radius=3.,
    lrange=None,results_true=False,interp_type="nearest",freqs=[0,1,2,3,4,5],
    n_inpaint=100,inpaint_flag=True,lsep=3000,beam="gaussian",
    theta_500_vec=np.exp(np.linspace(np.log(0.5),np.log(10.),10)),n_theta_refine=5,
    extract_at_truth=False,theta_500_input=None,norm_type="R_500",n_clusters_true=None,
    extraction_mode="find",save_cov=False,true_cat_select="q",cov_name=None,q_true_type="grid",
    subgrid_label=True,find_subgrid=False,theta_find="input",detection_method="maxima",
    subtract_tsz_spectra=False,apod_type="old",mask_other_clusters=False,path="/Users/user/Desktop/",
    rank=0,max_it=1,mmf_type="standard",sed_b=None,exp=None,cmmf_type="one_dep",a_matrix=None,
    comp_to_calculate=[0],profile_type="arnaud",it_min_for_fixed=0):

        if concentration is None:

            concentration = 1.177

        self.pix = pix
        self.theta_range = theta_range
        self.q_th = q_th
        self.concentration = concentration
        self.refinement = refinement
        self.it = it
        self.cosmology = cosmology
        self.q_th_noise = q_th_noise
        self.decouple = decouple
        self.estimate_spec = estimate_spec
        self.fac = fac
        self.save_mask = save_mask
        self.cm_compute_scratch = cm_compute_scratch
        self.decouple_type = decouple_type
        self.mask_radius = mask_radius
        self.lrange = lrange
        self.interp_type = interp_type
        self.freqs = freqs
        self.n_inpaint = n_inpaint
        self.inpaint_flag = inpaint_flag
        self.lsep = lsep
        self.beam = beam
        self.theta_500_vec = theta_500_vec
        self.n_theta = len(self.theta_500_vec)
        self.n_theta_refine = n_theta_refine
        self.extract_at_truth = extract_at_truth
        self.theta_500_input = theta_500_input
        self.norm_type = norm_type
        self.n_clusters_true = n_clusters_true
        self.extraction_mode = extraction_mode
        self.save_cov = save_cov
        self.true_cat_select = true_cat_select
        self.cov_name = cov_name
        self.q_true_type = q_true_type
        self.subgrid_label = subgrid_label
        self.find_subgrid = find_subgrid
        self.theta_find = theta_find
        self.detection_method = detection_method
        self.subtract_tsz_spectra = subtract_tsz_spectra
        self.apod_type = apod_type
        self.mask_other_clusters = mask_other_clusters
        self.path = path
        self.rank = rank
        self.max_it = max_it
        self.mmf_type = mmf_type
        self.sed_b = sed_b
        self.exp = exp
        self.cmmf_type = cmmf_type
        self.a_matrix = a_matrix
        self.comp_to_calculate = comp_to_calculate
        self.profile_type = profile_type
        self.it_min_for_fixed = it_min_for_fixed

        self.info = {"pix":self.pix,"theta_range":self.theta_range,"q_th":self.q_th,
        "concentration":self.concentration,"refinement":self.refinement,
        "it":self.it,"cosmology":self.cosmology,"q_th_noise":self.q_th_noise,
        "decouple":self.decouple,"estimate_spec":self.estimate_spec,"fac":self.fac,
        "save_mask":self.save_mask,"cm_compute_scratch":self.cm_compute_scratch,
        "decouple_type":self.decouple_type,"mask_radius":self.mask_radius,
        "lrange":self.lrange,"interp_type":self.interp_type,"freqs":self.freqs,
        "n_inpaint":self.n_inpaint,"inpaint_flag":self.inpaint_flag,"lsep":self.lsep,
        "beam":self.beam,"theta_500_vec":self.theta_500_vec,"norm_type":self.norm_type,
        "n_clusters_true":self.n_clusters_true,"extraction_mode":self.extraction_mode,
        "save_cov":self.save_cov,"true_cat_select":self.true_cat_select,
        "cov_name":self.cov_name,"q_true_type":self.q_true_type,
        "subgrid_label":self.subgrid_label,"find_subgrid":self.find_subgrid,
        "theta_find":self.theta_find,"detection_method":self.detection_method,
        "subtract_tsz_spectra":self.subtract_tsz_spectra,"apod_type":self.apod_type,
        "mask_other_clusters":self.mask_other_clusters,"rank":self.rank,
        "mask_it":self.max_it,"mmf_type":self.mmf_type,"sed_b":self.sed_b,"exp":self.exp,
        "cmmf_type":self.cmmf_type,"a_matrix":self.a_matrix,
        "comp_to_calculate":self.comp_to_calculate,"profile_type":self.profile_type,
        "it_min_for_fixed":self.it_min_for_fixed}

    def find_clusters(self,t_obs,mask_map=None,t_noise=None,pixel_id=0,mask_select=None,mask_select_no_tile=None,
    mask_ps=None,save_name=None,save_maps=False,mask_point=None,true_catalogue=None,mask_peak_finding_no_tile=None,
    mask_peak_finding=None,t_true=None,fixed_catalogue=None):

        if t_true is None:

            t_true = t_obs

        if mask_map is None:

            mask_map = np.ones((self.pix.nx,self.pix.ny))

        if mask_peak_finding is None:

            mask_peak_finding = mask_map

        if mask_peak_finding_no_tile is None:

            mask_peak_finding_no_tile = mask_peak_finding

        if mask_select is None:

            mask_select = mask_map

        if mask_ps is None:

            mask_ps = mask_map

        if mask_point is None:

            mask_point = np.ones((self.pix.nx,self.pix.ny))

        if mask_select_no_tile is None:

            mask_select_no_tile = mask_select

    #    if self.extract_at_truth == True:

    #        self.theta_500_vec = true_catalogue.theta_500
    #        self.refinement = False

        if fixed_catalogue is None:

            fixed_catalogue = true_catalogue

        if self.a_matrix is None:

            self.a_matrix = self.exp.tsz_f_nu[self.freqs]

        self.t_obs = t_obs
        self.t_true = t_true
        self.mask_map = mask_map
        self.t_noise = t_noise
        self.pixel_id = pixel_id
        self.mask_select = mask_select
        self.mask_point = mask_point
        self.mask_ps = mask_ps
        self.mask_select_no_tile = mask_select_no_tile
        self.mask_peak_finding = mask_peak_finding
        self.mask_peak_finding_no_tile = mask_peak_finding_no_tile
        self.save_name = save_name
        self.save_maps = save_maps
        self.true_catalogue = true_catalogue
        self.fixed_catalogue = fixed_catalogue

        self.t_obs = maps.select_freqs(self.t_obs,self.freqs)
        self.t_true = maps.select_freqs(self.t_true,self.freqs)
        self.t_noise = maps.select_freqs(self.t_noise,self.freqs)

        if t_noise is None:

            self.t_noise = self.t_obs

        if self.inpaint_flag == True:

            self.t_obs_inp = inpaint.diffusive_inpaint_freq(self.t_obs,self.mask_point,self.n_inpaint)

        else:

            self.t_obs_inp = self.t_obs

        mask_ps_0 = self.mask_ps
        mask_point_0 = self.mask_point

        i = 0

        self.results = cat.results_detection()
        self.results.info = self.info
        self.results.theta_500_sigma = self.theta_500_vec
        self.results.fsky = np.sum(self.mask_select)*self.pix.dx*self.pix.dy/(4.*np.pi)

        mask_cluster = np.ones((self.pix.nx,self.pix.ny))
        clusters_masked_old = cat.cluster_catalogue()

        #Apodise (except t_noise, which is apodised in the power spectra estimation functions)

        if self.apod_type == "new":

            mask_map_t = maps.clone_map_freq(self.mask_map,self.t_obs.shape[2])
            self.t_obs_inp = self.t_obs_inp*mask_map_t
            self.t_true = self.t_true*mask_map_t

        self.t_obs_inp = maps.filter_tmap(self.t_obs_inp,self.pix,self.lrange)
        self.t_noise = maps.filter_tmap(self.t_noise,self.pix,self.lrange)
        self.t_true = maps.filter_tmap(self.t_true,self.pix,self.lrange)

        while i >= 0:

            if self.rank == 0:

                print("Noise it",i)

            #self.results = cat.results_detection(results_grid_noit=self.results.results_grid_noit,
            #results_refined_noit=self.results.results_refined_noit)

            if np.array_equal(self.mask_select,np.zeros((self.pix.nx,self.pix.ny))):

                break

            #Inpaint t_noise

            self.mask_point = mask_point_0*mask_cluster

            if self.inpaint_flag == True:

                self.t_noise_inp = inpaint.diffusive_inpaint_freq(self.t_noise,self.mask_point,self.n_inpaint)

            else:

                self.t_noise_inp = self.t_noise

            self.t_noise_inp = maps.filter_tmap(self.t_noise_inp,self.pix,self.lrange)

            #Estimate spectra

            if self.estimate_spec == "estimate":

                if self.decouple_type == "master":

                    if np.array_equal(self.mask_ps,maps.get_apodised_mask(self.pix,np.ones((self.pix.nx,self.pix.ny)),
                    apotype="Smooth",aposcale=0.2)):

                        mask_name = "apod_smooth_1024.fits"
                        self.ps = spec.power_spectrum(self.pix,mask=self.mask_ps,cm_compute=True,cm_compute_scratch=False,
                        fac=self.fac,cm_save=self.save_mask,cm_name=mask_name)

                    else:

                        mask_name = self.path + "coupling_matrices_planck/apod_smooth_" + str(self.pixel_id) + ".fits"
                        self.ps = spec.power_spectrum(self.pix,mask=self.mask_ps,cm_compute=True,cm_compute_scratch=self.cm_compute_scratch,
                        fac=self.fac,cm_save=self.save_mask,cm_name=mask_name) #with real data change to name below

                    #np.save(save_name + "_mask_" + str(i) + ".npy",mask_ps)
                    #np.save("maps/sim_maps/binary.fits",mask_ps)

                else:

                    self.ps = spec.power_spectrum(self.pix,mask=self.mask_ps,cm_compute=False,fac=self.fac)

                self.cspec = spec.cross_spec(np.arange(len(self.freqs)))

                self.cspec.get_cross_spec(self.pix,t_map=self.t_noise_inp,ps=self.ps,decouple_type=self.decouple_type,
                inpaint_flag=self.inpaint_flag,mask_point=self.mask_point,lsep=self.lsep)
                self.inv_cov = self.cspec.get_inv_cov(self.pix,interp_type=self.interp_type)

                if self.save_cov == True:

                    np.save("/Users/user/Desktop/maps/websky_maps_injected/snr_maps_test/planck_cov_cmmf_" + str(self.pixel_id) + ".npy",self.inv_cov)

            elif self.estimate_spec == "theory":

                self.inv_cov = spec.cross_spec(self.freqs).get_inv_cov(self.pix,theory=True,cmb=True)

            elif self.estimate_spec == "load":

                if self.rank == 0:

                    print("cov name",self.cov_name)

                self.inv_cov = np.load(self.cov_name)

            self.inv_cov = maps.filter_cov(self.inv_cov,self.pix,self.lrange)

            #np.save("inverse_cov_test_5.npy",self.inv_cov)

            #Precompute weights for constrained MMF

            if self.mmf_type == "standard":

                self.cmmf_type = "standard_mmf"

            self.cmmf = cmmf_precomputation(self.pix,self.freqs,self.inv_cov,
            lrange=self.lrange,beam_type=self.beam,exp=self.exp,cmmf_type=self.cmmf_type,a_matrix=self.a_matrix,
            comp_to_calculate=self.comp_to_calculate)

            #SNR extraction

            if self.theta_find == "true":

                #won't work for websky

                true_catalogue_selected = select_true_catalogue(self.true_catalogue,self.theta_500_vec,self.results.sigma_vec,n_clusters=self.n_clusters_true)

                self.theta_500_vec = true_catalogue_selected.theta_500

            self.filtered_maps = filter_maps(self.t_obs_inp,self.inv_cov,self.pix,self.cosmology,
            self.theta_500_vec,self.q_th,theta_range=self.theta_range,pixel_id=self.pixel_id,
            save_maps=self.save_maps,save_name=self.save_name,i_it=i,mask_map=self.mask_map,mask_select_list=[self.mask_select,self.mask_select_no_tile],
            mask_peak_finding_list=[self.mask_peak_finding,self.mask_peak_finding_no_tile],
            lrange=self.lrange,freqs=self.freqs,beam=self.beam,norm_type=self.norm_type,find_subgrid=self.find_subgrid,
            detection_method=self.detection_method,apod_type=self.apod_type,path=self.path,rank=self.rank,
            mmf_type=self.mmf_type,cmmf=self.cmmf,exp=self.exp,profile_type=self.profile_type)

            if self.extraction_mode == "find" or (self.extraction_mode == "fixed" and self.true_cat_select == "q") or (self.extraction_mode == "fixed" and self.it == True and i < self.max_it):

                if self.rank == 0:

                    print("Cluster finding")

                self.filtered_maps.find_for_thetas()
                self.results.sigma_vec  = self.filtered_maps.sigma_vec

                if i == 0:

                    self.results.sigma_vec_noit  = self.filtered_maps.sigma_vec

                results_list = self.filtered_maps.results_list

                self.results.catalogues["catalogue_find_" + str(i)] = results_list[0]
                results_for_masking = results_list[1]

                if self.rank == 0:

                    print("q finding",np.flip(np.sort(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"])))

            if (self.extraction_mode == "fixed") and (i >= self.it_min_for_fixed):

                if self.rank == 0:

                    print("Extraction at fixed catalogue")

                true_catalogue_selected = select_true_catalogue(self.fixed_catalogue,self.theta_500_vec,
                self.results.sigma_vec,n_clusters=self.n_clusters_true)
                self.filtered_maps.extract_at_true_values(self.t_obs_inp,
                true_catalogue_selected,subgrid_label=self.subgrid_label,
                comp_to_calculate=self.comp_to_calculate,profile_type=self.profile_type)
                catalogue_obs = self.filtered_maps.catalogue_true_values

                self.results.catalogues["catalogue_fixed_" + str(i)] = catalogue_obs

                print("q fixed",np.flip(np.sort(self.results.catalogues["catalogue_fixed_" + str(i)].catalogue["q_opt"])))

            #Extract signal at true values

            if self.extract_at_truth == True:

                con1 = (self.it == True and i > 0)
                con2 = (self.it == False)
                con3 = (self.it == True and len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.)
                con4 = (self.it == True and all(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"] < self.q_th_noise))

                if (con1 or con2 or con3 or con4):

                    true_catalogue_selected = select_true_catalogue(self.true_catalogue,self.theta_500_vec,self.results.sigma_vec,n_clusters=self.n_clusters_true)

                    if self.q_true_type == "grid":

                        self.filtered_maps.extract_at_true_values(self.t_true,
                        true_catalogue_selected,subgrid_label=self.subgrid_label,
                        mask_other_clusters=self.mask_other_clusters,
                        comp_to_calculate=self.comp_to_calculate,profile_type=self.profile_type)
                        catalogue_true_values = self.filtered_maps.catalogue_true_values

                    elif self.q_true_type == "true":

                        catalogue_true_values = get_results_true(self.pix,self.inv_cov,true_catalogue_selected,
                        self.cosmology,self.pixel_id,freqs=self.freqs,beam=self.beam,theta_500_input=None,lrange=self.lrange,
                        norm_type=self.norm_type,path=self.path,cmmf_prec=self.cmmf,
                        exp=self.exp,profile_type=self.profile_type)

                    self.catalogue_true_values = catalogue_true_values
                    self.results.catalogues["catalogue_true_" + str(i)] = catalogue_true_values

                    print("q true",np.flip(np.sort(self.results.catalogues["catalogue_true_" + str(i)].catalogue["q_opt"])))

            if len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.:

                break

            self.n_clusters = len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"])

            if self.it == False:

                break

            if  self.estimate_spec == False:

                break

            if i == self.max_it:

                break

            #Cluster signal estimation and masking

            self.results_for_masking = results_for_masking

            clusters_masked_new = cat.apply_q_cut(self.results_for_masking,self.q_th_noise)
            clusters_masked_old = cat.apply_q_cut(clusters_masked_old,self.q_th_noise)

            if len(clusters_masked_new.catalogue["q_opt"]) == 0:

                break

            clusters_masked_old_id,clusters_masked_new_id = cat.identify_clusters(clusters_masked_old,clusters_masked_new)

            if np.all(clusters_masked_old_id.catalogue["q_opt"]) != -1 and i > 0:

                break

            mask_cluster = get_cluster_mask(self.pix,self.results_for_masking,self.q_th_noise,self.mask_radius)

            clusters_masked_old = clusters_masked_new

            i += 1

def get_cluster_mask(pix,catalogue,q_th_noise,mask_radius):

    mask_cluster = np.ones((pix.nx,pix.ny))

    for j in range(0,len(catalogue.catalogue["q_opt"])):

        if catalogue.catalogue["q_opt"][j] > q_th_noise:

            x_est,y_est = maps.get_theta_misc([catalogue.catalogue["theta_x"][j],catalogue.catalogue["theta_y"][j]],pix)
            source_coords = np.zeros((1,2))
            source_coords[0,0] = x_est
            source_coords[0,1] = y_est
            mask_cluster *= maps.ps_mask(pix,1,catalogue.catalogue["theta_500"][j]*mask_radius).get_mask_map(source_coords=source_coords)

    return mask_cluster

def get_cluster_mask_one(pix,catalogue,mask_radius):

    x_est,y_est = maps.get_theta_misc([catalogue.catalogue["theta_x"],catalogue.catalogue["theta_y"]],pix)
    source_coords = np.zeros((1,2))
    source_coords[0,0] = x_est
    source_coords[0,1] = y_est
    mask_cluster = maps.ps_mask(pix,1,catalogue.catalogue["theta_500"]*mask_radius).get_mask_map(source_coords=source_coords)

    return mask_cluster

def select_true_catalogue(true_catalogue,theta_500_vec,sigma_vec,n_clusters=10):

    if len(true_catalogue.catalogue["q_opt"]) > n_clusters:

        q_true = true_catalogue.catalogue["y0"]/np.interp(true_catalogue.catalogue["theta_500"],theta_500_vec,sigma_vec)
        indices = np.argpartition(q_true,-n_clusters)[-n_clusters:]
        ret = cat.get_catalogue_indices(true_catalogue,indices)

    else:

        ret = true_catalogue

    return ret


def get_results_true(pix,inv_cov,true_catalogue,cosmology,pixel_id,freqs=[0,1,2,3,4,5],
beam="gaussian",theta_500_input=None,lrange=[0,100000],norm_type="centre",path="/Users/user/Desktop/",
mmf_type="standard",cmmf_prec=None,exp=None,profile_type="arnaud"):

    theta_500_vec = true_catalogue.catalogue["theta_500"]
    theta_x_vec = true_catalogue.catalogue["theta_x"]
    theta_y_vec = true_catalogue.catalogue["theta_y"]
    z = true_catalogue.catalogue["z"]
    m_500 = true_catalogue.catalogue["m_500"]

    q_true = np.zeros(len(theta_500_vec))
    y0_true = np.zeros(len(theta_500_vec))

    for i in range(0,len(theta_500_vec)):

        #nfw = model.gnfw_tsz(model.get_m_500(theta_500_vec[i],z,cosmology),z,cosmology)
        nfw = model.gnfw_tsz(m_500[i]/1e15,z[i],cosmology,path=path,type=profile_type)
        t_cluster_centre,tem_uc_centre = nfw.get_t_map_convolved(pix,exp,beam=beam,get_nc=True,sed=False)

        if theta_500_input is not None:

            nfw_tem = model.gnfw_tsz(model.get_m_500(theta_500_input,z,cosmology),z,
            cosmology,path=path,type=profile_type)
            norm = nfw_tem.get_y_norm(norm_type)

            tem,tem_nc = nfw_tem.get_t_map_convolved(pix,exp,beam=beam,get_nc=True,sed=False)
            tem = tem/norm
            tem_nc = tem_nc/norm
            tem = maps.select_freqs(tem,freqs)
            tem_nc = maps.select_freqs(tem_nc,freqs)

        else:

            norm = nfw.get_y_norm(norm_type)
            tem = t_cluster_centre/norm
            tem = maps.select_freqs(tem,freqs)
            tem_nc = tem_nc_centre/norm
            tem_nc = aps.select_freqs(tem_nc,freqs)


        t_cluster_centre = maps.filter_tmap(t_cluster_centre,pix,lrange)
        tem = maps.filter_tmap(tem,pix,lrange)
        tem_nc = maps.filter_tmap(tem_nc,pix,lrange)

        q_map,y_map,std = get_mmf_q_map(t_cluster_centre,tem,inv_cov,pix,mmf_type=mmf_type,cmmf_prec=cmmf_prec,tem_nc=tem_nc)

        q_true[i] = np.max(q_map)
        y0_true[i] = norm

    cat_return = cat.cluster_catalogue()

    cat_return.catalogue["q_opt"] = q_true
    cat_return.catalogue["y0"] = y0_true
    cat_return.catalogue["theta_500"] = theta_500_vec
    cat_return.catalogue["theta_x"] = theta_x_vec
    cat_return.catalogue["theta_y"] = theta_y_vec
    cat_return.catalogue["pixel_ids"] = np.ones(len(q_true))*pixel_id
    cat_return.catalogue["m_500"] = m_500
    cat_return.catalogue["z"] = z

    return cat_return


class filter_maps:

    def __init__(self,t_obs,inv_cov,pix,cosmology,theta_500_vec,q_th,
    theta_range=None,pixel_id=0,save_maps=False,save_name=None,i_it=0,mask_map=None,
    mask_select_list=None,mask_peak_finding_list=None,lrange=None,freqs=[0,1,2,3,4,5],beam="gaussian",
    norm_type="centre",find_subgrid=False,theta_find="input",detection_method="maxima",mmf_type="standard",
    cmmf = None,
    apod_type="old",path="/Users/user/Desktop/",rank=0,exp=None,profile_type="arnaud"):

        if theta_range == None:

            theta_x_min = 0.
            theta_x_max = pix.nx*pix.dx
            theta_y_min = 0.
            theta_y_max = pix.ny*pix.dy

            theta_range = [theta_x_min,theta_x_max,theta_y_min,theta_y_max]

        if mask_map is None:

            mask_map = np.ones((pix.nx,pix.ny))

        if mask_select_list is None:

            mask_select_list = [mask_map]

        if mask_peak_finding_list is None:

            mask_peak_finding_list = [mask_map]

        self.t_obs = t_obs
        self.inv_cov = inv_cov
        self.pix = pix
        self.cosmology = cosmology
        self.theta_500_vec = theta_500_vec
        self.q_th = q_th
        self.theta_range = theta_range
        self.pixel_id = pixel_id
        self.save_maps = save_maps
        self.save_name = save_name
        self.i_it = i_it
        self.mask_map = mask_map
        self.mask_select_list = mask_select_list
        self.mask_peak_finding_list = mask_peak_finding_list
        self.lrange = lrange
        self.freqs = freqs
        self.beam = beam
        self.norm_type = norm_type
        self.find_subgrid = find_subgrid
        self.theta_find = theta_find
        self.detection_method = detection_method
        self.apod_type = apod_type
        self.path = path
        self.rank = rank
        self.mmf_type = mmf_type
        self.cmmf = cmmf
        self.exp = exp
        self.profile_type = profile_type

    def find_for_thetas(self):

        t_obs = self.t_obs

        if self.apod_type == "old":

            mask_map_t = maps.clone_map_freq(self.mask_map,t_obs.shape[2])
            t_obs = t_obs*mask_map_t

        n_theta = len(self.theta_500_vec)

        if self.find_subgrid == True:

            n_subgrid = 4

        elif self.find_subgrid == False:

            n_subgrid = 1

        subgrid_ix = [0.,0.,0.5,0.5]
        subgrid_jx = [0.,0.5,0.,0.5]

        self.q_tensor = np.zeros((self.pix.nx,self.pix.ny,n_theta,n_subgrid))
        self.y_tensor = np.zeros((self.pix.nx,self.pix.ny,n_theta,n_subgrid))
        self.sigma_vec = np.zeros(n_theta)

        for k in range(0,n_subgrid):

            if self.find_subgrid == True:

                if self.rank == 0:

                    print("Subgrid",k)

            for j in range(0,n_theta):


                if self.rank == 0:

                    print("Theta",j,self.theta_500_vec[j])

                if self.profile_type == "point":

                    ps = model.point_source(self.exp,beam_type=self.beam)
                    t_tem = ps.get_t_map_convolved(self.pix)
                    tem_nc = None

                else:

                    z = 0.2
                    M_500 = model.get_m_500(self.theta_500_vec[j],z,self.cosmology)
                    nfw = model.gnfw_tsz(M_500,z,self.cosmology,path=self.path,type=self.profile_type)

                    t_tem,t_tem_nc = nfw.get_t_map_convolved(self.pix,self.exp,beam=self.beam,
                    theta_cart=[(0.5*self.pix.nx+subgrid_ix[k])*self.pix.dx,(0.5*self.pix.nx+subgrid_jx[k])*self.pix.dx],
                    get_nc=True,sed=False)

                    t_tem = t_tem/nfw.get_y_norm(self.norm_type)

                    t_tem_nc = maps.select_freqs(t_tem_nc,self.freqs)
                    tem_nc = maps.filter_tmap(t_tem_nc,self.pix,self.lrange)/nfw.get_y_norm(self.norm_type)

                t_tem = maps.select_freqs(t_tem,self.freqs)
                tem = maps.filter_tmap(t_tem,self.pix,self.lrange)

                if self.apod_type == "old":

                    t_obs = maps.filter_tmap(t_obs,self.pix,self.lrange)

                q_map,y_map,std = get_mmf_q_map(t_obs,tem,self.inv_cov,self.pix,mmf_type=self.mmf_type,
                cmmf_prec=self.cmmf,tem_nc=tem_nc)

                if self.save_maps == True:

                    #pl.imshow(q_map*self.mask_select_list[0])
                    #pl.savefig(self.save_name + "_q_" + str(self.i_it) + "_" + str(j) + ".pdf")
                    #pl.close()
                    np.save(self.save_name + "_q_" + str(self.i_it) + "_" + str(j) + ".npy",q_map*self.mask_select_list[0])

                self.q_tensor[:,:,j,k] = q_map
                self.y_tensor[:,:,j,k] = y_map

                if k == 0:

                    self.sigma_vec[j] = std


        #pl.semilogx(self.theta_500_vec,self.sigma_vec)
        #pl.show()

        n_mask_select = len(self.mask_select_list)
        self.results_list = []

        x_coord = maps.rmap(self.pix).get_x_coord_map_wrt_origin() #vertical coordinate, in rad
        y_coord = maps.rmap(self.pix).get_y_coord_map_wrt_origin() #horizontal coordinate, in rad

        for i in range(0,n_mask_select):

            q_tensor = apply_mask_peak_finding(self.q_tensor,self.mask_peak_finding_list[i])
            y_tensor = apply_mask_peak_finding(self.y_tensor,self.mask_peak_finding_list[i])

            indices = make_detections(q_tensor,self.q_th,self.pix,detection_method=self.detection_method)

            q_opt = q_tensor[indices]
            y0_est = y_tensor[indices]
            x_est = x_coord[(indices[0],indices[1])]
            y_est = y_coord[(indices[0],indices[1])]
            theta_est = self.theta_500_vec[indices[2]]
            n_detect = len(q_opt)

            theta_x = y_est + self.theta_range[0]
            theta_y = self.pix.nx*self.pix.dx - x_est + self.theta_range[2]

            cat_new = cat.cluster_catalogue()

            cat_new.catalogue["q_opt"] = q_opt
            cat_new.catalogue["y0"] = y0_est
            cat_new.catalogue["theta_500"] = theta_est
            cat_new.catalogue["theta_x"] = theta_x
            cat_new.catalogue["theta_y"] = theta_y
            cat_new.catalogue["pixel_ids"] = np.ones(len(q_opt))*self.pixel_id

            cat_new = cat.apply_mask_select(cat_new,self.mask_select_list[i],self.pix)
            self.results_list.append(cat_new)

        return 0

    def extract_at_true_values(self,t_true,true_catalogue,subgrid_label=True,
    mask_other_clusters=False,comp_to_calculate=[0],profile_type="arnaud"):

        if self.apod_type == "old":

            mask_map_t = maps.clone_map_freq(self.mask_map,t_true.shape[2])
            t_true = t_true*mask_map_t


        n_clus = len(true_catalogue.catalogue["theta_x"])

        q_opt = np.zeros((n_clus,len(comp_to_calculate)))
        y0_est = np.zeros((n_clus,len(comp_to_calculate)))

        if self.rank == 0:

            print("Extracting at true parameters")

        t_true_unmasked = t_true

        for i in range(0,n_clus):

            if self.rank == 0:

                print("theta_500",true_catalogue.catalogue["theta_500"][i])

            z = 0.2 #true_catalogue.z[i]
            M_500 = model.get_m_500(true_catalogue.catalogue["theta_500"][i],z,self.cosmology)

            if mask_other_clusters == True:

                t_true = t_true_unmasked
                mask_radius = 10 #in units of theta_500
                cat_for_mask = cat.get_catalogue_indices(true_catalogue,i)
                mask_for_other_clusters = get_cluster_mask_one(self.pix,cat_for_mask,mask_radius)
                mask_for_other_clusters = maps.clone_map_freq(1.-mask_for_other_clusters,t_true.shape[2])
                t_true = t_true*mask_for_other_clusters

            q_extracted,y0_extracted,q_map = extract_at_input_value(t_true,self.inv_cov,self.pix,
            self.beam,M_500,z,self.cosmology,self.norm_type,
            true_catalogue.catalogue["theta_x"][i],true_catalogue.catalogue["theta_y"][i],
            self.lrange,subgrid_label=subgrid_label,apod_type=self.apod_type,path=self.path,
            mmf_type=self.mmf_type,cmmf_prec=self.cmmf,freqs=self.freqs,exp=self.exp,
            comp_to_calculate=comp_to_calculate,profile_type=profile_type)

            q_opt[i,:] = q_extracted
            y0_est[i,:] = y0_extracted

            if self.save_maps == True:

                #pl.imshow(q_map*self.mask_select_list[0])
                #pl.savefig(self.save_name + "_q_true_" + str(self.i_it) + "_" + str(i) + ".pdf")
                #pl.close()
                #np.save(self.save_name + "_q_true_" + str(self.i_it) + "_" + str(i) + ".npy",q_map*self.mask_select_list[0])
                np.save(self.save_name,q_map)

        catalogue_at_true_values = cat.cluster_catalogue()

        catalogue_at_true_values.catalogue["q_opt"] = q_opt[:,0]
        catalogue_at_true_values.catalogue["y0"] = y0_est[:,0]
        catalogue_at_true_values.catalogue["theta_500"] = true_catalogue.catalogue["theta_500"]
        catalogue_at_true_values.catalogue["theta_x"] = true_catalogue.catalogue["theta_x"]
        catalogue_at_true_values.catalogue["theta_y"] = true_catalogue.catalogue["theta_y"]
        catalogue_at_true_values.catalogue["pixel_ids"] =  np.ones(n_clus)*self.pixel_id
        catalogue_at_true_values.catalogue["m_500"] = true_catalogue.catalogue["m_500"]
        catalogue_at_true_values.catalogue["z"] = true_catalogue.catalogue["z"]

        #Extraction of other components

        for i in range(1,len(comp_to_calculate)):

            catalogue_at_true_values.catalogue["q_c" + str(i)] = q_opt[:,i]
            catalogue_at_true_values.catalogue["c" + str(i)] = y0_est[:,i]

        self.catalogue_true_values = cat.apply_mask_select(catalogue_at_true_values,self.mask_select_list[1],self.pix)


def extract_at_input_value(t_true,inv_cov,pix,beam,M_500,z,cosmology,norm_type,
theta_x,theta_y,lrange,subgrid_label="True",apod_type="old",path="/Users/user/Desktop/",
mmf_type="standard",cmmf_prec=None,freqs=[0,1,2,3,4,5],exp=None,comp_to_calculate=[0],
profile_type="arnaud"):

    nfw = model.gnfw_tsz(M_500,z,cosmology,path=path,type=profile_type)

    if apod_type == "old":

        t_true = maps.filter_tmap(t_true,pix,lrange)

        q_opt = np.zeros(len(comp_to_calculate))
        y0_est = np.zeros(len(comp_to_calculate))

    if subgrid_label == False:

        tem,tem_nc = nfw.get_t_map_convolved(pix,exp,beam=beam,get_nc=True,sed=False)
        tem = tem/nfw.get_y_norm(norm_type)
        tem_nc = tem_nc/nfw.get_y_norm(norm_type)
        tem = maps.select_freqs(tem,freqs)
        tem_nc = maps.select_freqs(tem_nc,freqs)
        tem = maps.filter_tmap(tem,pix,lrange)
        tem_nc = maps.filter_tmap(tem_nc,pix,lrange)

        for k in range(0,len(comp_to_calculate)):

            q_map_tem,y_map,std = get_mmf_q_map(t_true,tem,inv_cov,pix,mmf_type=mmf_type,
            cmmf_prec=cmmf_prec,tem_nc=tem_nc,comp=comp_to_calculate[k])

            if k == 0:

                q_map = q_map_tem

            j_clus = int(np.floor(theta_x/pix.dx))
            i_clus = int(pix.ny-np.ceil(theta_y/pix.dy))

            q_opt[k] = q_map_tem[i_clus,j_clus]
            y0_est[k] = y_map[i_clus,j_clus]

    elif subgrid_label == True:

        theta_cart_tem = [theta_x-np.floor(theta_x/pix.dx)*pix.dx+pix.nx*pix.dx/2,
        theta_y-np.floor(theta_y/pix.dx)*pix.dx+pix.nx*pix.dx/2]
        theta_cart = [theta_x,theta_y]

        tem,tem_nc = nfw.get_t_map_convolved(pix,exp,beam=beam,theta_cart=theta_cart_tem,get_nc=True,sed=False)
        tem = tem/nfw.get_y_norm(norm_type)
        tem_nc = tem_nc/nfw.get_y_norm(norm_type)
        tem = maps.select_freqs(tem,freqs)
        tem_nc = maps.select_freqs(tem_nc,freqs)
        tem = maps.filter_tmap(tem,pix,lrange)
        tem_nc = maps.filter_tmap(tem_nc,pix,lrange)

        for k in range(0,len(comp_to_calculate)):

            q_map_tem,y_map,std = get_mmf_q_map(t_true,tem,inv_cov,pix,mmf_type=mmf_type,
            cmmf_prec=cmmf_prec,tem_nc=tem_nc,comp=comp_to_calculate[k])

            if k == 0:

                q_map = q_map_tem

            dj = theta_cart[0]/pix.dx - np.floor(theta_cart[0]/pix.dx)
            di = theta_cart[1]/pix.dx - np.floor(theta_cart[1]/pix.dx)

            if 0 <= dj < 0.25:

                delta_j = 0

            elif 0.25 <= dj <= 0.75:

                delta_j = 1

            elif 1 > dj > 0.75:

                delta_j = 2

            if di < 0.25:

                delta_i = 0

            elif 0.25 <= di <= 0.75:

                delta_i = 1

            elif 1 > di > 0.75:

                delta_i = 2

            i,j = maps.get_ij_from_theta(np.floor(theta_cart[0]/pix.dx)*pix.dx,np.floor(theta_cart[1]/pix.dx)*pix.dx,pix)

            j_extract = j + delta_j
            i_extract = i - delta_i

            q_opt[k] = q_map_tem[i_extract,j_extract]
            y0_est[k] = y_map[i_extract,j_extract]

    return q_opt,y0_est,q_map

def apply_mask_peak_finding(tensor,mask):

    for i in range(0,tensor.shape[2]):

        for j in range(0,tensor.shape[3]):

            tensor[:,:,i,j] *= mask

    return tensor

def make_detections(q_tensor,q_th,pix,find_subgrid=False,detection_method="maxima"):

    indices = np.where(q_tensor > q_th)

    coords = np.zeros((len(indices[0]),2),dtype=int)
    coords[:,0] = indices[0]
    coords[:,1] = indices[1]
    theta_coord = np.array(indices[2])
    subgrid_coord = np.array(indices[3])

    if coords.shape[0] == 0:

        ret = ([],[],[],[])

    else:

        if detection_method == "DBSCAN":

            clust = DBSCAN(eps=2)
            clust.fit(coords)
            labels = clust.labels_
            n_clusters = np.max(labels)+1

            i_opt_vec = np.zeros(n_clusters,dtype=int)
            j_opt_vec = np.zeros(n_clusters,dtype=int)
            theta_opt_vec = np.zeros(n_clusters,dtype=int)
            subgrid_vec = np.zeros(n_clusters,dtype=int)

            for i in range(0,n_clusters):

                indices = np.where(labels==i)
                q_tensor_select = q_tensor[coords[:,0][indices],coords[:,1][indices],theta_coord[indices],subgrid_coord[indices]]
                opt_index = np.argmax(q_tensor_select)
                i_opt_vec[i]  = coords[:,0][indices][opt_index]
                j_opt_vec[i] = coords[:,1][indices][opt_index]
                theta_opt_vec[i] = theta_coord[indices][opt_index]
                subgrid_vec[i] = subgrid_coord[indices][opt_index]
                q_opt = q_tensor_select[opt_index]

            ret = (i_opt_vec,j_opt_vec,theta_opt_vec,subgrid_vec)

        elif detection_method == "maxima":

            q_tensor_maxima = q_tensor[:,:,:,0].copy()
            q_tensor_maxima[np.where(q_tensor_maxima < q_th)] = 0.

            maxs_idx = get_maxima(get_maxima_mask(q_tensor_maxima))

            i = np.array(maxs_idx[:,0]).astype(int)
            j = np.array(maxs_idx[:,1]).astype(int)
            theta = np.array(maxs_idx[:,2]).astype(int)
            subgrid = np.array(np.zeros(len(i))).astype(int)

            ret = (i,j,theta,subgrid)

    return ret

# Mascara booleana de si es un maximo o no, quitando al volumen 3D los bordes
def get_maxima_mask(r_input):

    r = np.ones((r_input.shape[0]+2,r_input.shape[1]+2,r_input.shape[2]+2)) * -np.inf
    r[1:-1,1:-1,1:-1] = r_input

    xmax=np.argmax([r[2:,1:-1,1:-1],r[:-2,1:-1,1:-1],r[1:-1,1:-1,1:-1]],axis=0) == 2
    ymax=np.argmax([r[1:-1,2:,1:-1], r[1:-1,:-2,1:-1],r[1:-1,1:-1,1:-1]], axis=0) == 2
    zmax=np.argmax([r[1:-1,1:-1, 2:], r[1:-1,1:-1, :-2], r[1:-1,1:-1, 1:-1]], axis=0) == 2

    return xmax&ymax&zmax

# Indices de que posiciones en el volumen original son maximas

def get_maxima(max_mask):

    maxima = np.argwhere(max_mask == True)

    return maxima

def get_theta_500_refined(results,n_theta_refine,theta_500_vec):

    theta_est_vec = np.unique(results.results_grid.catalogue["theta_500"])
    theta_500_refined = np.zeros(0)

    for i in range(0,len(theta_est_vec)):

        index = np.where(theta_500_vec == theta_est_vec[i])[0]

        if index == 0:

            theta_min = theta_500_vec[index]
            theta_max = theta_500_vec[index+1]
            n_ref = n_theta_refine//2

        elif index == len(theta_500_vec)-1:

            theta_min = theta_500_vec[index-1]
            theta_max = theta_500_vec[index]
            n_ref = n_theta_refine//2

        else:

            theta_min = theta_500_vec[index-1]
            theta_max = theta_500_vec[index+1]
            n_ref = n_theta_refine

        theta_500_refined = np.append(theta_500_refined,np.linspace(theta_min,theta_max,n_ref))

    return theta_500_refined


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


def get_matched_filter(rec,tem,noi,pix):

    norm_estimate, norm_variance = filter_sum(maps.get_fft(rec,pix),maps.get_fft(tem,pix),noi)

    return (norm_estimate,norm_variance)

def get_matched_filter_real(rec,tem,noi):

    map1 = rec*tem/noi
    map2 = tem**2/noi

    sum_1 = np.sum(map1)
    sum_2 = np.sum(map2)

    normalisation_estimate = sum_1/sum_2
    normalisation_variance = 1./sum_2

    return (normalisation_estimate,normalisation_variance)

def get_mf_map(obs,tem,noi,pix):

    tem_fft = maps.get_fft(tem,pix)
    tem_conv = maps.get_ifft(tem_fft/noi,pix).real

    map_convolution = sg.fftconvolve(obs,tem_conv,mode='same')*pix.dx*pix.dy
    norm = np.max(sg.fftconvolve(tem,tem_conv,mode='same'))*pix.dx*pix.dy
    est_map = map_convolution/norm
    std = 1./np.sqrt(norm)

    return est_map,std

def div0(a,b):

    with np.errstate(divide='ignore',invalid='ignore'):

        c = np.true_divide(a,b)
        c[~np.isfinite(c)] = 0

    return c


def get_mmf_q_map(tmap,tem,inv_cov,pix,theta_misc_template=[0.,0.],
mmf_type="standard",cmmf_prec=None,tem_nc=None,comp=0):

    if mmf_type == "partially_spectrally_constrained":

        a = 1.

    else:

        q_map,y_map,std = get_mmf_q_map_s(tmap,tem,inv_cov,pix,
        theta_misc_template=theta_misc_template,mmf_type=mmf_type,cmmf_prec=cmmf_prec,
        tem_nc=tem_nc,comp=comp)


    return q_map,y_map,std

def get_mmf_q_map_s(tmap,tem,inv_cov,pix,theta_misc_template=[0.,0.],
mmf_type="standard",cmmf_prec=None,tem_nc=None,comp=0):

    n_freqs = tmap.shape[2]
    y_map = np.zeros((pix.nx,pix.ny))
    norm_map = np.zeros((pix.nx,pix.ny))

    tem_fft = maps.get_fft_f(tem,pix)

    if tem_nc is not None: #Used if cmmf_type == "general"

        tem_nc = maps.get_fft_f(tem_nc,pix)

        """

        if cmmf_prec.cmmf_type == "general":

            tem_nc = maps.get_tmap_times_fvec(tem_nc,1./cmmf_prec.a_matrix[:,0])
            tem_fft = maps.get_tmap_times_fvec(tem_fft,1./cmmf_prec.a_matrix[:,0])
        """

    filter_fft = get_tem_conv_fft(pix,tem_fft,inv_cov,mmf_type=mmf_type,cmmf_prec=cmmf_prec,
    tem_nc=tem_nc,comp=comp)
    filter = maps.get_ifft_f(filter_fft,pix).real

    tem = maps.get_tmap_times_fvec(tem,cmmf_prec.a_matrix[:,comp]).real #new line

    for i in range(n_freqs):

        y = sg.fftconvolve(tmap[:,:,i],filter[:,:,i],mode='same')*pix.dx*pix.dy
        norm = sg.fftconvolve(tem[:,:,i],filter[:,:,i],mode='same')*pix.dx*pix.dy

        y_map += y
        norm_map += norm

    #norm = norm_map[pix.nx//2,pix.nx//2]
    norm = np.max(norm_map)
    y_map = y_map/norm

#    if mmf_type == "standard":
    std = 1./np.sqrt(norm)

    #elif mmf_type == "spectrally_constrained":

    #std2 = get_cmmf_std(pix,mmf_type,inv_cov,filter_fft,norm)

    #print("std comparison",std,std2)

    q_map = y_map/std

    return q_map,y_map,std


def get_mmf_centre(tmap,tem,inv_cov,pix,freqs=[0,1,2,3,4,5],mmf_type="standard",cmmf_prec=None):

    n_freqs = tmap.shape[2]
    y_map = np.zeros((pix.nx,pix.ny))
    norm_map = np.zeros((pix.nx,pix.ny))

    tem_fft = maps.get_fft_f(tem,pix)
    filter_fft = get_tem_conv_fft(pix,tem_fft,inv_cov,mmf_type=mmf_type,cmmf_prec=cmmf_prec)
    tmap_fft = maps.get_fft_f(tmap,pix)

    y_0 = 0
    norm = 0
    y_map = np.zeros((pix.nx,pix.ny))

    for i in range(len(freqs)):

        y_i = np.sum(np.conjugate(filter_fft[:,:,i])*tmap_fft[:,:,i])
        norm_i = np.sum(np.conjugate(filter_fft[:,:,i])*tem_fft[:,:,i])

        y_map = y_map + np.conjugate(filter_fft[:,:,i])*tmap_fft[:,:,i]

        y_0 = y_0 + y_i
        norm = norm + norm_i

    y_0 = y_0.real
    norm = norm.real

    y_0 = y_0/norm

    if mmf_type == "standard":

        std = 1./np.sqrt(norm)

    elif mmf_type == "spectrally_constrained":

        std = get_cmmf_std(pix,mmf_type,inv_cov,filter_fft,freqs,norm)

    return y_0,norm,std

def get_mmf_centre_direct(tmap,tem,inv_cov,pix,freqs=[0,1,2,3,4,5],mmf_type="standard",cmmf_prec=None):

    tem_fft = maps.get_fft_f(tem,pix)
    tmap_fft = maps.get_fft_f(tmap,pix)

    filter_fft = get_tem_conv_fft(pix,tem_fft,inv_cov,mmf_type=mmf_type,cmmf_prec=cmmf_prec)

    n_freq = tem_fft.shape[2]
    a_dot_b =  maps.get_tmap_from_map(cmmf_prec.a_dot_b,n_freq)
    b_dot_b = maps.get_tmap_from_map(cmmf_prec.b_dot_b,n_freq)
    b = cmmf_prec.sed_b
    a = cmmf_prec.sed_a

    a_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
    a_map_fft = maps.get_tmap_times_fvec(a_map_fft,a)
    b_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
    b_map_fft = maps.get_tmap_times_fvec(b_map_fft,b)

    tem_fft_b = maps.get_tmap_times_fvec(maps.get_tmap_times_fvec(tem_fft,1./a),b)

    tem_fft_no_sed = maps.get_tmap_times_fvec(tem_fft,1./a)
    tmap_fft_no_sed = maps.get_tmap_times_fvec(tmap_fft,1./b)

    map = get_inv_cov_dot(np.conjugate(tem_fft),inv_cov,tmap_fft) - a_dot_b[:,:,0]/b_dot_b[:,:,0]*get_inv_cov_dot(np.conjugate(tem_fft_b),inv_cov,tmap_fft)
    norm = get_inv_cov_dot(np.conjugate(tem_fft),inv_cov,tem_fft) - a_dot_b[:,:,0]/b_dot_b[:,:,0]*get_inv_cov_dot(np.conjugate(tem_fft_b),inv_cov,tmap_fft)

    #alternative

    #map = (get_inv_cov_dot(a_map_fft,inv_cov,b_map_fft) - a_dot_b[:,:,0]/b_dot_b[:,:,0]*get_inv_cov_dot(b_map_fft,inv_cov,b_map_fft))*np.conjugate(tem_fft_no_sed[:,:,0])*tmap_fft_no_sed[:,:,0]

    #map = (get_inv_cov_dot(np.conjugate(tmap_fft),inv_cov,tmap_fft)
    #- a_dot_b[:,:,0]/b_dot_b[:,:,0]*get_inv_cov_dot(np.conjugate(tem_fft_b),inv_cov,tmap_fft))

    #print((maps.get_tmap_times_fvec(maps.get_tmap_times_fvec(tem_fft,1./a),a)/tem_fft))

#    map = (get_inv_cov_dot(np.conjugate(tem_fft),inv_cov,tmap_fft)
#    - a_dot_b[:,:,0]/b_dot_b[:,:,0]*get_inv_cov_dot(np.conjugate(tem_fft_b),inv_cov,tmap_fft))

    #norm = (get_inv_cov_dot(a_map_fft,inv_cov,a_map_fft) - a_dot_b[:,:,0]/b_dot_b[:,:,0]*get_inv_cov_dot(b_map_fft,inv_cov,a_map_fft))*np.conjugate(tem_fft_no_sed[:,:,0])*tem_fft_no_sed[:,:,0]
#    norm = (get_inv_cov_dot(np.conjugate(tem_fft),inv_cov,tem_fft) - a_dot_b[:,:,0]/b_dot_b[:,:,0]*get_inv_cov_dot(np.conjugate(tem_fft_b),inv_cov,tem_fft))#*np.conjugate(tem_fft_no_sed[:,:,0])*tmap_fft_no_sed[:,:,0]

    #norm = (get_inv_cov_dot(np.conjugate(tmap_fft),inv_cov,tem_fft)
    #- a_dot_b[:,:,0]/b_dot_b[:,:,0]*get_inv_cov_dot(np.conjugate(tem_fft_b),inv_cov,tem_fft))


    y_0 = np.sum(map)
    norm = np.sum(norm)

    y_0 = y_0.real
    norm = norm.real

    y_0 = y_0/norm

#    if mmf_type == "standard":

#        std = 1./np.sqrt(norm)

#    elif mmf_type == "spectrally_constrained":

    std = get_cmmf_std(pix,mmf_type,inv_cov,filter_fft,freqs,norm)

    return y_0,norm,std

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

                inv_cov[i,j,:,:] = np.linalg.inv(cov[i,j,:,:])

    return inv_cov


def get_tem_conv_fft(pix_tem,tem_fft,inv_cov,mmf_type="standard",cmmf_prec=None,
tem_nc=None,comp=0):

    if mmf_type == "standard":

        tem_fft = maps.get_tmap_times_fvec(tem_fft,cmmf_prec.a_matrix[:,comp]) #new line
        filter_fft = get_inv_cov_conjugate(tem_fft,inv_cov)

    elif mmf_type == "spectrally_constrained": #doesn't admit changing compoenent order

        if cmmf_prec.cmmf_type == "one_dep":

            n_freq = tem_fft.shape[2]
            a_dot_b =  maps.get_tmap_from_map(cmmf_prec.a_dot_b,n_freq)
            b_dot_b = maps.get_tmap_from_map(cmmf_prec.b_dot_b,n_freq)
            b = cmmf_prec.a_matrix[:,1] #cmmf_prec.sed_b
            a = cmmf_prec.a_matrix[:,0] #cmmf_prec.sed_a

            tem_fft_b = maps.get_tmap_times_fvec(tem_fft,b)
            tem_fft = maps.get_tmap_times_fvec(tem_fft,a)

            filter_fft = get_inv_cov_conjugate(tem_fft,inv_cov) - a_dot_b/b_dot_b*get_inv_cov_conjugate(tem_fft_b,inv_cov)

        elif cmmf_prec.cmmf_type == "general":

            filter_fft_0 = cmmf_prec.w_norm_list[comp]
            filter_fft = filter_fft_0*tem_nc

    filter_fft[np.isnan(filter_fft)] = 0.

    return filter_fft

def get_inv_cov_conjugate(tem_fft,inv_cov):

    return np.einsum('dhi,dhij->dhj',tem_fft,inv_cov)

def get_inv_cov_dot(a,inv_cov,b):

    conjugate = get_inv_cov_conjugate(a,inv_cov)

    return np.einsum('dhi,dhi->dh',conjugate,b)


def get_mf_q_map(tmap,tem,inv_cov,pix_map,pix_tem=None):

    if pix_tem == None:

        pix_tem = pix_map

    y_map = np.zeros((pix_map.nx,pix_map.ny))
    norm_map = np.zeros((pix_tem.nx,pix_tem.ny))

    tem_fft = maps.get_fft(tem,pix_tem)

    tem_conv_fft = tem_fft*inv_cov
    tem_conv = maps.get_ifft(tem_conv_fft,pix_tem).real

    y_map = sg.fftconvolve(tmap,tem_conv,mode='same')*pix_map.dx*pix_map.dy
    norm_map = sg.fftconvolve(tem,tem_conv,mode='same')*pix_map.dx*pix_map.dy

    norm = norm_map[pix_map.nx//2,pix_map.nx//2]
    y_map = y_map/norm
    std = 1/np.sqrt(norm)
    q_map = y_map/std

    return q_map,y_map,std

def get_mf_q_map_investigate(tmap,tem,inv_cov,pix):

    y_map = np.zeros((pix.nx,pix.ny))
    norm_map = np.zeros((pix.nx,pix.ny))

    tem_fft = maps.get_fft(tem,pix)

    tem_conv_fft = tem_fft*inv_cov
    tem_conv = maps.get_ifft(tem_conv_fft,pix).real

    y_map = sg.fftconvolve(tmap,tem_conv,mode='same')*pix.dx*pix.dy
    norm_map = sg.fftconvolve(tem,tem_conv,mode='same')*pix.dx*pix.dy

    norm = norm_map[pix.nx//2,pix.nx//2]
    y_map = y_map/norm
    std = 1/np.sqrt(norm)
    q_map = y_map/std

    return q_map,y_map,std

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

    """
    for i in range(0,pix_tem.nx):

        for j in range(0,pix_tem.ny):

            tem_conv_fft[i,j,:] = np.matmul(tem_fft[i,j,:],inv_cov[i,j,:,:])

    """

    filter_fft = get_tem_conv_fft(pix,tem_fft,inv_cov,mmf_type=mmf_type,cmmf_prec=cmmf_prec)
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

def get_q_opt_gaussian(pix,obs,tem,inv_cov,ell_range=None):

    if ell_range == None:

        ell_range = [0,10**5]

    nx = pix.nx

    """
    cov = cspec.spec_tensor
    ell = cspec.ell_vec
    fac = ell*(ell+1.)/(2.*np.pi)
    pl.plot(ell,cov[:,0,0])
    pl.show()
    """

    #MMF

    #q_map,y_map,std = get_mmf_q_map(obs,tem,inv_cov,pix,freqs=[0])

    #q_map,y_map,std = get_mf_q_map_investigate(obs[:,:,0],tem[:,:,0],inv_cov[:,:,0,0],pix)
    norm_est,norm_var = get_matched_filter(obs[:,:,0],tem[:,:,0],1./inv_cov[:,:,0,0],pix)
    q_opt = norm_est/np.sqrt(norm_var)
    #q_opt = norm_est

    return q_opt,norm_est

def get_q_bias(pix,tem,signal,cov_mf_true,cov_noi_true,n_modes_per_bin_map,type="q"):

    #pl.imshow(n_modes_per_bin_map)
    #pl.show()

    tem_fft = maps.get_fft(tem,pix)
    signal_fft = maps.get_fft(signal,pix)

    n0 = filter_sum_1(tem_fft,tem_fft,cov_mf_true)

    map1 = (signal_fft+np.conjugate(signal_fft))*cov_noi_true*np.conjugate(tem_fft)
    map2 = (signal_fft+np.conjugate(signal_fft))*cov_noi_true*np.conjugate(tem_fft)*np.abs(tem_fft)**2
    map1 = div0(map1,cov_mf_true**2*n_modes_per_bin_map)
    map2 = div0(map2,cov_mf_true**3*n_modes_per_bin_map)

    sum1 = np.sum(map1).real
    sum2 = np.sum(map2).real

    #sum1 = filter_sum_1((signal_fft+np.conjugate(signal_fft))*cov_noi_true,tem_fft,cov_mf_true**2)
    #sum2 = filter_sum_1((signal_fft+np.conjugate(signal_fft))*cov_noi_true,tem_fft*np.abs(tem_fft)**2,cov_mf_true**2)

    if type == "q":

        bias1 = -sum1/n0**0.5#*0.5
        bias2 = 0.5*sum2/n0**1.5

    elif type == "y0":

        bias1 = -sum1/n0
        bias2 = sum2/n0**2

    #bias1 = -sum1/n0
    #bias2 = sum2/n0**2

    #map1 = np.conjugate(tem_fft)*signal_fft/cov_mf_true
    #map1 = div0(np.conjugate(tem_fft)*signal_fft,cov_mf_true)


    #q_mean = np.sum(map1)/np.sqrt(n0)
    #A = np.sum(map1).real/n0

    #print("A mean",A)

    return bias1,bias2

def get_q_std_bias(pix,tem,signal,cov_mf_true,cov_noi_true):

    tem_fft = maps.get_fft(tem,pix)
    signal_fft = maps.get_fft(signal,pix)
    n0 = filter_sum_1(tem_fft,tem_fft,cov_mf_true)
    n_bias = filter_sum_1(tem_fft,tem_fft*cov_noi_true,(cov_mf_true)**2)
    n_bias = filter_sum_1(tem_fft,tem_fft*cov_noi_true,(cov_mf_true)**2)
    sigma_unbiased = 1./np.sqrt(n0)
    sigma_biased = 1./np.sqrt(n_bias)

    return sigma_unbiased/sigma_biased


def filter_sum_1(rec,tem,noi):

    map1 = np.conjugate(tem)*rec
    map1 = div0(map1,noi)

    return np.sum(map1).real

class inv_cov_gaussian:

    def __init__(self,pix,obs,implementation="custom"):

        nx = pix.nx
        mask = np.ones((nx,nx))
        ps = spec.power_spectrum(pix,mask=mask,cm_compute=False,fac=4.)

        cspec = spec.cross_spec([0])
        cspec.get_cross_spec(pix,t_map=obs,ps=ps,decouple_type="none",
        inpaint_flag=False,implementation=implementation)

        self.cl = cspec.spec_tensor[:,0,0]
        self.ell = cspec.ell_vec
        self.ell_min = cspec.ps.l0_bins[0]
        self.ell_max = cspec.ps.lf_bins[-1]

        if implementation == "custom":

            self.n_modes_per_bin = cspec.ps.n_modes_per_bin
            self.ell_map = cspec.ps.ell_map
            ell = cspec.ps.ell_eff

            #self.n_modes_per_bin_map = interpolate.interp1d(ell,self.n_modes_per_bin,
            #kind="nearest",bounds_error=False,fill_value="extrapolate")(self.ell_map)
            self.n_modes_per_bin_map = cspec.ps.n_modes_per_bin_map


            ell_map_masked = self.ell_map.flatten()
            indices = np.where((ell_map_masked < self.ell_min) | (ell_map_masked > self.ell_max))
            ell_map_masked[indices] = 0.
            ell_map_masked = ell_map_masked.reshape(self.ell_map.shape)

            #pl.imshow(ell_map_masked)
            #pl.show()
            #pl.imshow(self.n_modes_per_bin_map)
            #pl.show()

        factor = self.ell*(self.ell+1.)**2/(2.*np.pi)

        #pl.plot(ell,cl)
        #pl.show()

        self.inv_cov = cspec.get_inv_cov(pix,interp_type="nearest")

def get_map_convolved_fft(map_fft_original,pix,freqs,beam_type,mask,lrange,exp):

    a_map_fft = maps.convolve_tmap_fft_experiment(pix,map_fft_original,exp,freqs,beam_type=beam_type)
    a_map_fft = maps.get_fft_f(maps.get_ifft_f(a_map_fft,pix)*mask,pix)
    a_map_fft = maps.filter_fft_f(a_map_fft,pix,lrange)

    return a_map_fft

class cmmf_precomputation:

    #sed's in muK

    def __init__(self,pix,freqs,inv_cov,lrange=[0,1000000],beam_type="gaussian",exp=None,
    mask=None,cmmf_type="one_dep",a_matrix=None,n_dep=1,comp_to_calculate=[0]):

        self.cmmf_type = cmmf_type
        self.a_matrix = a_matrix

        if mask is None:

            mask = np.ones((pix.nx,pix.ny,len(freqs)))

        if self.cmmf_type == "one_dep":

            self.sed_a = a_matrix[:,0][freqs] #exp.tsz_f_nu[freqs]
            self.sed_b = a_matrix[:,1][freqs] #sed_b[freqs]

            a_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
            a_map_fft = maps.get_tmap_times_fvec(a_map_fft,self.sed_a)

            a_map_fft = get_map_convolved_fft(a_map_fft,pix,freqs,beam_type,mask,lrange,exp)

            """
            a_map_fft = maps.convolve_tmap_fft_experiment(pix,a_map_fft,exp,freqs,beam_type=beam_type)
            a_map_fft = maps.get_fft_f(maps.get_ifft_f(a_map_fft,pix)*mask,pix)
            a_map_fft = maps.filter_fft_f(a_map_fft,pix,lrange)
            """

            b_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
            b_map_fft = maps.get_tmap_times_fvec(b_map_fft,self.sed_b)

            b_map_fft = get_map_convolved_fft(b_map_fft,pix,freqs,beam_type,mask,lrange,exp)

            """
            b_map_fft = maps.convolve_tmap_fft_experiment(pix,b_map_fft,exp,freqs,beam_type=beam_type)
            b_map_fft = maps.get_fft_f(maps.get_ifft_f(b_map_fft,pix)*mask,pix)
            b_map_fft = maps.filter_fft_f(b_map_fft,pix,lrange)
            """

            self.a_dot_b = get_inv_cov_dot(np.conjugate(a_map_fft),inv_cov,b_map_fft)
            self.b_dot_b = get_inv_cov_dot(np.conjugate(b_map_fft),inv_cov,b_map_fft)

        elif self.cmmf_type == "general":

            #a_matrix is f x c matrix (f number of frequencies, c number of components, 1st SZ, rest to deproject)


            a_matrix_fft = np.zeros((pix.nx,pix.nx,len(freqs),self.a_matrix.shape[1]))

            for i in range(0,a_matrix.shape[1]):

                a_matrix_fft_i = np.ones((pix.nx,pix.nx,len(freqs)))
                a_matrix_fft_i = maps.get_tmap_times_fvec(a_matrix_fft_i,self.a_matrix[freqs,i])
                a_matrix_fft[:,:,:,i] = get_map_convolved_fft(a_matrix_fft_i,pix,freqs,beam_type,mask,lrange,exp)
                #a_matrix_fft[:,:,:,i] = a_matrix_fft_i

            self.a_matrix_fft = a_matrix_fft

            pre = np.einsum('abcd,abde->abce',inv_cov,a_matrix_fft)
            dot = invert_cov(np.einsum('abdc,abde->abce',a_matrix_fft,pre))
            self.W = np.einsum('abcd,abed->abce',dot,pre)
            cov = invert_cov(inv_cov)

            t0 = time.time()

            self.w_list = []
            self.sigma_y_list = []
            self.w_norm_list = []

            for i in range(0,len(comp_to_calculate)):

                e_map = np.zeros((pix.nx,pix.nx,self.a_matrix.shape[1]))
                e_map[:,:,comp_to_calculate[i]] = 1.

                w = np.einsum('ija,ijab->ijb',e_map,self.W)

                sigma_y = get_inv_cov_dot(np.conjugate(w),cov,w)

                w_norm = np.zeros(w.shape)

                for i in range(0,len(freqs)):

                    w_norm[:,:,i] = w[:,:,i]/sigma_y

                self.w_list.append(w)
                self.sigma_y_list.append(sigma_y)
                self.w_norm_list.append(w_norm)

        elif self.cmmf_type == "standard_mmf":

            r = 1.
