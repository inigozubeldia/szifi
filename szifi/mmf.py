import numpy as np
import scipy.signal as sg
from szifi import params, maps, model, cat, spec, utils, sed
import warnings, os
warnings.filterwarnings("ignore")
import gzip

#Core class for cluster finding
class cluster_finder:

    def __init__(self,params_szifi=params.params_szifi_default,
    params_model=params.params_model_default,
    data_file=None,
    rank=0):

        self.params_szifi = params_szifi
        self.params_model = params_model
        self.data_file = data_file.data #instance of input_data
        self.theta_500_vec = self.params_szifi["theta_500_vec_arcmin"]
        self.rank = rank
        self.exp = self.data_file["experiment"]
        self.cosmology = model.cosmological_model(self.params_szifi).cosmology
        self.indices_filter=None
        if self.params_szifi["detection_method"] == "DBSCAN":
            from sklearn.cluster import DBSCAN

        if self.params_szifi["mmf_type"] == "standard" and self.params_szifi["a_matrix"] is None:
            a_matrix = np.zeros((len(self.exp.tsz_sed),1))
            a_matrix[:,0] = self.exp.tsz_sed
            self.params_szifi["a_matrix"] = a_matrix

        if self.params_szifi["deproject_cib"] is not None:
            get_a_matrix_cib(self.params_szifi,self.params_model,self.data_file)

    def find_clusters(self):
        #Print some information
        if self.params_szifi["mmf_type"] == "standard":
            type_name = "standard"

        elif self.params_szifi["mmf_type"] == "spectrally_constrained":
            n_dep = self.params_szifi["a_matrix"].shape[1]-1
            type_name = "spectrally constrained, " + str(n_dep) + " component(s) deprojected"

        print("SZiFi")
        print("")
        print("")
        print("MMF type:",type_name)
        print("Iterative:",self.params_szifi["iterative"])
        print("Extraction mode:",self.params_szifi["extraction_mode"])
        print("Experiment:",self.exp.experiment_name)
        print("Frequency channels:",self.params_szifi["freqs"])

        #Iterate over fields

        self.results_dict = {}

        for field_id in self.data_file["params_data"]["field_ids"]:

            print("")
            print("")
            print("Field",field_id)
            print("")

            #Gather input data
            map_dtype = self.params_szifi["map_dtype"]
            self.t_obs = utils.extract(self.data_file, "t_obs", field_id, map_dtype) #in muK
            self.t_noi = utils.extract(self.data_file, "t_noi", field_id, map_dtype) #in muK

            self.mask_point = utils.extract(self.data_file, "mask_point", field_id, map_dtype)
            self.mask_ps = utils.extract(self.data_file, "mask_ps", field_id, map_dtype)
            self.mask_select = utils.extract(self.data_file, "mask_select", field_id, map_dtype)
            self.mask_select_no_tile = utils.extract(self.data_file, "mask_select_no_tile", field_id, map_dtype)
            self.mask_map = utils.extract(self.data_file, "mask_map", field_id, map_dtype)
            self.mask_peak_finding_no_tile = utils.extract(self.data_file, "mask_peak_finding_no_tile", field_id, map_dtype)
            self.mask_peak_finding = utils.extract(self.data_file, "mask_peak_finding", field_id, map_dtype)

            self.dx = self.data_file["dx_arcmin"][field_id]/60./180.*np.pi
            self.nx = self.data_file["nx"][field_id]
            self.pix = maps.pixel(self.nx,self.dx)

            if self.params_szifi["decouple_type"] == "master" and self.params_szifi["compute_coupling_matrix"] == False:
                self.coupling_matrix_name = self.data_file["coupling_matrix_name"][field_id]

            else:
                self.coupling_matrix_name = None
            if self.params_szifi["extraction_mode"] == "fixed":
                self.catalogue_fixed = self.data_file["catalogue_input"][field_id]

            #Select frequency channels to use
            self.t_obs = maps.select_freqs(self.t_obs,self.params_szifi["freqs"])
            self.t_noi = maps.select_freqs(self.t_noi,self.params_szifi["freqs"])
            if self.params_szifi["get_q_true"] == True:
                self.t_true = utils.extract(self.data_file, "t_true", field_id, map_dtype)
                self.t_true = maps.select_freqs(self.t_true,self.params_szifi["freqs"])

            #Inpaint point sources
            if self.params_szifi["inpaint"] == True:

                self.t_obs = maps.diffusive_inpaint_freq(self.t_obs,self.mask_point,self.params_szifi["n_inpaint"])

            # mask_ps_0 = self.mask_ps
            mask_point_0 = self.mask_point

            #Initialise results class

            self.results = cat.results_detection()
            self.results.info = self.params_szifi
            self.results.theta_500_vec = self.theta_500_vec
            self.results.fsky = np.sum(self.mask_select)*self.pix.dx*self.pix.dy/(4.*np.pi)

            #Initialise cluster masking (if iterative noise covariance estimation is required)

            mask_cluster = np.ones((self.pix.nx,self.pix.ny), dtype=self.t_obs.dtype)
            clusters_masked_old = cat.cluster_catalogue()

            #Apodise (except t_noise, which is apodised in the power spectra estimation functions)

            if self.params_szifi["apod_type"] == "new":

                self.t_obs = maps.multiply_t(self.mask_map, self.t_obs)

                if self.params_szifi["get_q_true"] == True:

                    self.t_true = maps.multiply_t(self.mask_map, self.t_true)

            #Filter input maps in harmonic space

            [lmin,lmax] = self.params_szifi['lrange']
            self.indices_filter = tuple(np.asarray(np.where((maps.rmap(self.pix).get_ell() < lmin) |  (maps.rmap(self.pix).get_ell() > lmax)), dtype=np.int32))
            self.t_obs = maps.filter_tmap(self.t_obs,self.pix,self.params_szifi["lrange"], indices_filter=self.indices_filter)

            self.t_noi = maps.filter_tmap(self.t_noi,self.pix,self.params_szifi["lrange"], indices_filter=self.indices_filter)

            t_noi_original = self.t_noi
            if self.params_szifi["get_q_true"] == True:

                self.t_true = maps.filter_tmap(self.t_true,self.pix,self.params_szifi["lrange"], indices_filter=self.indices_filter)

            #Initiate loop over iterative noise covariance estimation

            i = 0

            while i >= 0:
                if self.rank == 0:

                    print("Noise it",i)

                if not np.any(self.mask_select):
                    print(f"{self.rank}: mask_select all zeroes for field {field_id}, iteration {i}; breaking")
                    break

                self.mask_point = mask_point_0*mask_cluster

                #Inpaint noise map (as it may change with iteration)
                if self.params_szifi["inpaint"] == True:

                    self.t_noi = maps.diffusive_inpaint_freq(t_noi_original,self.mask_point,self.params_szifi["n_inpaint"])

                self.t_noi = maps.filter_tmap(self.t_noi,self.pix,self.params_szifi["lrange"], indices_filter=self.indices_filter)
                #Estimate channel cross-spectra

                if self.params_szifi["estimate_spec"] == "estimate":

                    lmax1d = self.params_szifi["powspec_lmax1d"]
                    new_shape = self.params_szifi["powspec_new_shape"]

                    if lmax1d is not None:

                        if new_shape is not None:

                            raise ValueError("Only one of powspec_lmax1d or powspec_new_shape can be specified")

                        new_shape = maps.get_newshape_lmax1d((self.pix.nx, self.pix.ny), lmax1d, self.pix.dx)

                    bin_fac = self.params_szifi["powspec_bin_fac"]
                    self.ps = spec.power_spectrum(self.pix,
                    mask=self.mask_ps,
                    cm_compute=True,
                    cm_compute_scratch=self.params_szifi["compute_coupling_matrix"],
                    cm_save=self.params_szifi["save_coupling_matrix"],
                    cm_name=self.coupling_matrix_name,
                    bin_fac=bin_fac,
                    new_shape=new_shape)

                    self.cspec = spec.cross_spec(np.arange(len(self.params_szifi["freqs"])))

                    self.cspec.get_cross_spec(self.pix,
                    t_map=self.t_noi,
                    ps=self.ps,
                    decouple_type=self.params_szifi["decouple_type"],
                    inpaint_flag=self.params_szifi["inpaint"],
                    mask_point=self.mask_point,
                    lsep=self.params_szifi["lsep"],
                    bin_fac=bin_fac,
                    new_shape=new_shape)

                    self.inv_cov = self.cspec.get_inv_cov(self.pix,interp_type=self.params_szifi["interp_type"], bin_fac=bin_fac, new_shape=new_shape)

                    #self.inv_cov_exp = maps.expand_matrix(self.inv_cov, (self.pix.nx, self.pix.ny))

                #If power spectrum is theoretically predicted - testing not up to date, do not use

                elif self.params_szifi["estimate_spec"] == "theory":

                    self.inv_cov = spec.cross_spec(self.params_szifi["freqs"]).get_inv_cov(self.pix,theory=True,cmb=True)


                del self.mask_point
                #Filter covariance matrix

                #self.inv_cov = maps.filter_cov(self.inv_cov,self.pix,self.params_szifi["lrange"])

                #Compute weights for constrained MMF

                if self.params_szifi["mmf_type"] == "standard":

                    self.params_szifi["cmmf_type"] = "standard_mmf"


                self.cmmf = scmmf_precomputation(pix=self.pix,
                freqs=self.params_szifi["freqs"],
                inv_cov=self.inv_cov,
                lrange=self.params_szifi["lrange"],
                beam_type=self.params_szifi["beam"],
                exp=self.exp,
                cmmf_type=self.params_szifi["cmmf_type"],
                a_matrix=self.params_szifi["a_matrix"],
                comp_to_calculate=self.params_szifi["comp_to_calculate"],
                mmf_type=self.params_szifi["mmf_type"])

                #Matched filter construction

                self.filtered_maps = filter_maps(t_obs=self.t_obs,
                inv_cov=self.inv_cov,
                pix=self.pix,
                cosmology=self.cosmology,
                theta_500_vec=self.theta_500_vec,
                field_id=field_id,
                i_it=i,
                params=self.params_szifi,
                params_model=self.params_model,
                mask_map=self.mask_map,
                mask_select_dict={"tile": self.mask_select, "field": self.mask_select_no_tile},
                mask_peak_finding_dict={"tile": self.mask_peak_finding, "field": self.mask_peak_finding_no_tile},
                rank=self.rank,
                exp=self.exp,
                cmmf=self.cmmf)

                #SZiFi in cluster finding mode: blind cluster detection

                if self.params_szifi["extraction_mode"] == "find" or (self.params_szifi["extraction_mode"] == "fixed" and self.params_szifi["iterative"] == True):# and i < self.max_it):

                    if self.rank == 0:

                        print("Cluster finding")

                    self.filtered_maps.find_clusters()

                    self.results.sigma_vec["find_" + str(i)] = self.filtered_maps.sigma_vec

                    self.results.catalogues["catalogue_find_" + str(i)] = self.filtered_maps.results['tile']
                    results_for_masking = self.filtered_maps.results['field']

                    if self.rank == 0:

                        print("Detections SNR",np.flip(np.sort(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"])))
                #SZiFi in fixed mode: extraction for an input catalogue

                if (self.params_szifi["extraction_mode"] == "fixed"):# and (i >= self.it_min_for_fixed):

                    con1 = (self.params_szifi["iterative"] == True and i > 0)
                    con2 = (self.params_szifi["iterative"] == False)
                    con3 = (self.params_szifi["iterative"] == True and len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.)
                    con4 = (self.params_szifi["iterative"] == True and all(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"] < self.params_szifi["q_th_noise"]))

                    if (con1 or con2 or con3 or con4):

                        if self.rank == 0:

                            print("Extraction for input catalogue")

                        self.filtered_maps.extract_at_true_values(self.t_obs,
                        self.catalogue_fixed)
                        catalogue_obs = self.filtered_maps.catalogue_true_values

                        self.results.catalogues["catalogue_fixed_" + str(i)] = catalogue_obs

                        print("Fixed SNR",np.flip(np.sort(self.results.catalogues["catalogue_fixed_" + str(i)].catalogue["q_opt"])))

                #Extract true SNR

                if self.params_szifi["get_q_true"] == True:

                    con1 = (self.params_szifi["iterative"] == True and i > 0)
                    con2 = (self.params_szifi["iterative"] == False)
                    con3 = (self.params_szifi["iterative"] == True and len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.)
                    con4 = (self.params_szifi["iterative"] == True and all(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"] < self.params_szifi["q_th_noise"]))

                    if (con1 or con2 or con3 or con4):

                        self.filtered_maps.extract_at_true_values(self.t_true,
                        self.catalogue_fixed,
                        comp_to_calculate=self.params_szifi["comp_to_calculate"],
                        profile_type=self.params_model["profile_type"])
                        catalogue_true_values = self.filtered_maps.catalogue_true_values

                        self.catalogue_true_values = catalogue_true_values
                        self.results.catalogues["catalogue_true_" + str(i)] = catalogue_true_values

                        print("q true",np.flip(np.sort(self.results.catalogues["catalogue_true_" + str(i)].catalogue["q_opt"])))

                #Several breaking conditions

                if len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.:

                    break

                self.n_clusters = len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"])

                if self.params_szifi["iterative"] == False:

                    break

                if  self.params_szifi["estimate_spec"] == False:

                    break

                if i == self.params_szifi["max_it"]:

                    break

                #Cluster masking for iterative covariance estimation

                self.results_for_masking = results_for_masking

                clusters_masked_new = cat.apply_q_cut(self.results_for_masking,self.params_szifi["q_th_noise"])
                clusters_masked_old = cat.apply_q_cut(clusters_masked_old,self.params_szifi["q_th_noise"])

                if len(clusters_masked_new.catalogue["q_opt"]) == 0:

                    break

                clusters_masked_old_id,clusters_masked_new_id = cat.identify_clusters(clusters_masked_old,clusters_masked_new)

                if np.all(clusters_masked_old_id.catalogue["q_opt"]) != -1 and i > 0:

                    break

                mask_cluster = get_cluster_mask(self.pix,self.results_for_masking,
                self.params_szifi["q_th_noise"],self.params_szifi["mask_radius"])

                clusters_masked_old = clusters_masked_new

                i += 1

            # Clean up the saved templates
            if self.params_szifi["save_and_load_template"] and hasattr(self, 'filtered_maps'):
                for j_theta in range(len(self.theta_500_vec)):
                    os.remove(self.filtered_maps.template_name % j_theta)
                    os.remove(self.filtered_maps.template_norm_name % j_theta)

            if self.params_szifi["get_lonlat"] == True:

                self.results.get_lonlat(field_id,self.pix,nside=self.data_file["nside_tile"])


            self.results_dict[field_id] = self.results


#Get mask at the location of detected clusters (for iterative noise covariance estimation)

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

#Construct and apply MMF

class filter_maps:

    def __init__(self,t_obs=None,inv_cov=None,pix=None,cosmology=None,theta_500_vec=None,
    params=None,field_id=0,i_it=0,mask_map=None,mask_select_dict=None,
    mask_peak_finding_dict=None,rank=0,exp=None,cmmf=None,params_model=None,indices_filter=None):

        self.t_obs = t_obs
        self.inv_cov = inv_cov
        self.pix = pix
        self.cosmology = cosmology
        self.theta_500_vec = theta_500_vec
        self.params = params
        self.params_model = params_model
        self.field_id = field_id
        self.i_it = i_it
        self.mask_map = mask_map
        self.mask_select_dict = mask_select_dict
        self.mask_peak_finding_dict = mask_peak_finding_dict
        self.mask_names = ["field", "tile"] # Order of calculation and easy iteration
        self.rank = rank
        self.theta_500_vec = theta_500_vec
        self.exp = exp
        self.cmmf = cmmf
        self.indices_filter=indices_filter

        self.theta_range = [0.,pix.nx*pix.dx,0.,pix.ny*pix.dy]

        if self.params["save_and_load_template"]:
            self.template_name = self.params["path_template"] + f"tem_{self.field_id}_%s.npy" # % j (theta index)
            self.template_norm_name = self.params["path_template"] + f"tem_norm_{self.field_id}_%s.npy"

    #Find detections blindly

    def find_clusters(self):

        t_obs = self.t_obs
        if self.params["apod_type"] == "old":

            t_obs = maps.multiply_t(self.mask_map, t_obs)
            t_obs = maps.filter_tmap(t_obs,self.pix,self.params["lrange"], indices_filter=self.indices_filter)

        n_theta = len(self.theta_500_vec)
        detect_peaks_maxima = (self.params["detection_method"] == "maxima") and (n_theta > 3)
        if detect_peaks_maxima:
            self.peak_info = {}
            for mask_name in self.mask_names:
                self.peak_info[mask_name] = np.empty((5, 0), dtype=np.float32)
            self.q_tensor = np.zeros((self.pix.nx,self.pix.ny,3), dtype=self.params['map_dtype'])
            self.y_map_last = None
        else:
            self.q_tensor = np.zeros((self.pix.nx,self.pix.ny,n_theta), dtype=self.params['map_dtype'])
            self.y_tensor = np.zeros((self.pix.nx,self.pix.ny,n_theta), dtype=self.params['map_dtype'])
        self.sigma_vec = np.zeros(n_theta)

        for j in range(0,n_theta):

            if self.rank == 0:

                print("Theta",j,self.theta_500_vec[j])

            if self.params["save_and_load_template"] == True:
                template_name = self.template_name % j
                template_norm_name = self.template_norm_name % j
                if self.i_it == 0 and os.path.isfile(template_name):
                    raise RuntimeError(f"template file {template_name} exists already; overwriting not allowed to prevent collisions during parallel runs")

            if self.params["save_and_load_template"] == False or (self.params["save_and_load_template"] == True and self.i_it == 0):

                if self.params_model["profile_type"] == "point":

                    ps = model.point_source(self.exp,beam_type=self.params["beam"])
                    t_tem = ps.get_t_map_convolved(self.pix)
                    t_tem_norm = None

                elif self.params_model["profile_type"] == "arnaud":
                    z = 0.2
                    M_500 = model.get_m_500(self.theta_500_vec[j],z,self.cosmology)
                    nfw = model.gnfw(M_500,z,self.cosmology,
                    type=self.params_model["profile_type"])

                    theta_cart = [(0.5*self.pix.nx)*self.pix.dx,
                    (0.5*self.pix.nx)*self.pix.dx]

                    if self.params["mmf_type"] == "standard" or (self.params["mmf_type"] == "spectrally_constrained" and self.params["cmmf_type"] == "one_dep"):

                        t_tem = nfw.get_t_map_convolved(self.pix,
                        self.exp,
                        beam=self.params["beam"],
                        theta_cart=theta_cart,
                        get_nc=False,
                        sed=False)
                        t_tem_norm = None


                    elif self.params["mmf_type"] == "spectrally_constrained" and self.params["cmmf_type"] == "general":

                        #theta_misc = maps.get_theta_misc(theta_cart,self.pix)

                        t_tem_norm,t_tem = nfw.get_t_map_convolved(self.pix,
                        self.exp,
                        beam=self.params["beam"],
                        theta_cart=theta_cart,
                        get_nc=True,
                        sed=False)

                        t_tem_norm = t_tem_norm/nfw.get_y_norm(self.params["norm_type"])
                        t_tem_norm = maps.filter_tmap(t_tem_norm,self.pix,self.params["lrange"], indices_filter=self.indices_filter)

                    t_tem = t_tem/nfw.get_y_norm(self.params["norm_type"])

                tem = maps.filter_tmap(t_tem,self.pix,self.params["lrange"], indices_filter=self.indices_filter)

                if self.params["save_and_load_template"] == True:
                    np.save(template_name,tem)
                    np.save(template_norm_name,t_tem_norm)

                    print("Template saved")

            elif self.params["save_and_load_template"] == True and self.i_it > 0:

                tem = np.load(template_name,allow_pickle=True)[()]
                t_tem_norm = np.load(template_norm_name,allow_pickle=True)[()]

                print("Template loaded")

            q_map,y_map,std = get_mmf_q_map(t_obs,tem,self.inv_cov,self.pix,mmf_type=self.params["mmf_type"],
            cmmf_prec=self.cmmf,tem_norm=t_tem_norm)
            del tem, t_tem_norm

            if self.params["save_snr_maps"] == True:
                fil = gzip.GzipFile(self.params["snr_maps_path"] + "/" + self.params["snr_maps_name"] + "_q_" + str(self.i_it) + "_" + str(j) + ".npy.gz", 'w')
                #np.save(self.params["snr_maps_path"] + "/" + self.params["snr_maps_name"] + "_q_" + str(self.i_it) + "_" + str(j) + ".npy",q_map*self.mask_select_dict['tile'])
                np.save(file=fil, arr=q_map*self.mask_select_dict['tile'])
                fil.close()

            if detect_peaks_maxima: # For "maxima" method we do peak-finding here to save memory
                q_th = self.params['q_th']
                if j == 0:
                    self.q_tensor[:,:,1] = q_map
                    self.y_map_last = y_map
                    continue

                else:
                    self.q_tensor[:,:,2] = q_map

                    qm = self.q_tensor[:,:,1] # We actually want to use the previous one
                    zmax = np.argmax(self.q_tensor, axis=2) == 1
                    for mask_name in self.mask_names:
                        q_opt, y0_est, maxs_idx = make_detections2(qm, self.y_map_last, self.mask_peak_finding_dict[mask_name], q_th, zmax)

                        self.save_detections(q_opt, y0_est, maxs_idx, mask_name, j-1)

                    if j == n_theta-1:
                        qm = self.q_tensor[:,:,2]
                        zmax = np.argmax(self.q_tensor[:,:,1:], axis=2) == 1
                        for mask_name in self.mask_names:
                            q_opt, y0_est, maxs_idx = make_detections2(qm, y_map, self.mask_peak_finding_dict[mask_name], q_th, zmax)
                            self.save_detections(q_opt, y0_est, maxs_idx, mask_name, j)

                    self.q_tensor = np.roll(self.q_tensor, -1, axis=2)
                    self.y_map_last = y_map
                    del qm, zmax, q_map, y_map

            else:
                self.q_tensor[:,:,j] = q_map
                self.y_tensor[:,:,j] = y_map
                del q_map, y_map

            self.sigma_vec[j] = std

        self.results = {}

        x_coord = maps.rmap(self.pix).get_x_coord_map_wrt_origin() #vertical coordinate, in rad

        y_coord = maps.rmap(self.pix).get_y_coord_map_wrt_origin() #horizontal coordinate, in rad

        for mask_name in self.mask_names:

            if detect_peaks_maxima:
                q_opt, y0_est, theta_est, inds0, inds1 = self.peak_info[mask_name]
                inds0, inds1 = inds0.astype(np.int64), inds1.astype(np.int64)
                x_est = x_coord[(inds0, inds1)]
                y_est = y_coord[(inds0, inds1)]

            else:
                # deepcopy(self.q_tensor) here would be safest, otherwise self.q_tensor is modified in place
                # But for memory savings, this is OK if the order is (no_tile, tile)
                if self.mask_names != ['field', 'tile']:
                    raise ValueError("masks must be ['field', 'tile']")
                q_tensor = apply_mask_peak_finding(self.q_tensor, self.mask_peak_finding_dict[mask_name])
                indices = make_detections(q_tensor,self.params["q_th"],self.pix,detection_method=self.params["detection_method"])

                q_opt = q_tensor[indices]
                y_tensor = apply_mask_peak_finding(self.y_tensor,self.mask_peak_finding_dict[mask_name])
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
            cat_new.catalogue["pixel_ids"] = np.ones(len(q_opt))*self.field_id

            cat_new = cat.apply_mask_select(cat_new,self.mask_select_dict[mask_name],self.pix)
            self.results[mask_name] = cat_new

        return 0

    #Apply MMF for input catalogue

    def extract_at_true_values(self,t_true,true_catalogue):

        if self.params["apod_type"] == "old":

            t_true = maps.multiply_t(self.mask_map, t_true)
            t_true = maps.filter_tmap(t_true,self.pix,self.params['lrange'],indices_filter=self.indices_filter)

        n_clus = len(true_catalogue.catalogue["theta_x"])

        q_opt = np.zeros((n_clus,len(self.params["comp_to_calculate"])))
        y0_est = np.zeros((n_clus,len(self.params["comp_to_calculate"])))

        if self.rank == 0:

            print("Extracting at true parameters")

        t_true_unmasked = t_true

        cmmf = self.cmmf
        #cmmf_base = copy.deepcopy(cmmf)

        for i in range(0,n_clus):

            z = 0.2 #true_catalogue.z[i]
            M_500 = model.get_m_500(true_catalogue.catalogue["theta_500"][i],z,self.cosmology)

            T = None

            if self.params["rSZ"] == True:

                tsz_sed_model = sed.tsz_model(T_e=true_catalogue.catalogue["T"][i])

                if self.params["integrate_bandpass"] == True:

                    tsz_sed = tsz_sed_model.get_sed_exp_bandpass(self.exp)

                elif self.params["integrate_bandpass"] == False:

                    tsz_sed = tsz_sed_model.get_sed(self.exp.nu_eff)

                tsz_sed = tsz_sed[self.params["freqs"]]

                a_matrix = self.params["a_matrix"][self.params["freqs"],:]
                a_matrix[:,0] = tsz_sed

                cmmf = scmmf_precomputation(pix=self.pix,
                freqs=self.params["freqs"],
                inv_cov=self.inv_cov,
                lrange=self.params["lrange"],
                beam_type=self.params["beam"],
                exp=self.exp,
                cmmf_type=self.params["cmmf_type"],
                a_matrix=a_matrix,
                comp_to_calculate=self.params["comp_to_calculate"],
                mmf_type=self.params["mmf_type"])

            q_extracted,y0_extracted,q_map = extract_at_input_value(t_true,
            self.inv_cov,
            self.pix,
            self.params["beam"],
            M_500,
            z,
            self.cosmology,
            self.params["norm_type"],
            true_catalogue.catalogue["theta_x"][i],
            true_catalogue.catalogue["theta_y"][i],
            self.params["lrange"],
            apod_type=self.params["apod_type"],
            mmf_type=self.params["mmf_type"],
            cmmf_type=self.params["cmmf_type"],
            cmmf_prec=self.cmmf,
            freqs=self.params["freqs"],
            exp=self.exp,
            comp_to_calculate=self.params["comp_to_calculate"],
            profile_type=self.params_model["profile_type"],
            T=T,
            )

            q_opt[i,:] = q_extracted
            y0_est[i,:] = y0_extracted

            if self.rank == 0:

                print("theta_500",true_catalogue.catalogue["theta_500"][i],"SNR",q_extracted)

        catalogue_at_true_values = cat.cluster_catalogue()

        catalogue_at_true_values.catalogue["q_opt"] = q_opt[:,0]
        catalogue_at_true_values.catalogue["y0"] = y0_est[:,0]
        catalogue_at_true_values.catalogue["theta_500"] = true_catalogue.catalogue["theta_500"]
        catalogue_at_true_values.catalogue["theta_x"] = true_catalogue.catalogue["theta_x"]
        catalogue_at_true_values.catalogue["theta_y"] = true_catalogue.catalogue["theta_y"]
        catalogue_at_true_values.catalogue["pixel_ids"] =  np.ones(n_clus)*self.field_id
        catalogue_at_true_values.catalogue["m_500"] = true_catalogue.catalogue["m_500"]
        catalogue_at_true_values.catalogue["z"] = true_catalogue.catalogue["z"]

        #Extraction of other components

        for i in range(1,len(self.params["comp_to_calculate"])):

            catalogue_at_true_values.catalogue["q_c" + str(i)] = q_opt[:,i]
            catalogue_at_true_values.catalogue["c" + str(i)] = y0_est[:,i]

        self.catalogue_true_values = catalogue_at_true_values


    def save_detections(self, q_opt, y0_est, maxs_idx, mask_name, j_theta):

        thetas = np.ones_like(q_opt) * self.theta_500_vec[j_theta]
        ans = np.vstack([q_opt, y0_est, thetas, maxs_idx.T])
        self.peak_info[mask_name] = np.hstack([self.peak_info[mask_name], ans])


#Apply MMF for input catalogue

def extract_at_input_value(t_true,inv_cov,pix,beam,M_500,z,cosmology,norm_type,
theta_x,theta_y,lrange,apod_type=None,mmf_type=None,cmmf_prec=None,cmmf_type=None,
freqs=None,exp=None,comp_to_calculate=None,profile_type=None, indices_filter=None):


    nfw = model.gnfw(M_500,z,cosmology,type=profile_type)

    q_opt = np.zeros(len(comp_to_calculate))
    y0_est = np.zeros(len(comp_to_calculate))

    if mmf_type == "standard" or (mmf_type == "spectrally_constrained" and cmmf_type == "one_dep"):

        tem = nfw.get_t_map_convolved(pix,
        exp,
        beam=beam,
        get_nc=False,
        sed=False)
        t_tem_norm = None

    elif mmf_type == "spectrally_constrained" and cmmf_type == "general":

        t_tem_norm,tem = nfw.get_t_map_convolved(pix,
        exp,
        beam=beam,
        get_nc=True,
        sed=False)

        t_tem_norm = t_tem_norm/nfw.get_y_norm(norm_type)
        t_tem_norm = maps.filter_tmap(t_tem_norm,pix,lrange)

    tem = tem/nfw.get_y_norm(norm_type)
    tem = maps.select_freqs(tem,freqs)
    tem = maps.filter_tmap(tem,pix,lrange,indices_filter=indices_filter)

    for k in range(0,len(comp_to_calculate)):

        q_map_tem,y_map,std = get_mmf_q_map(t_true,tem,inv_cov,pix,mmf_type=mmf_type,
        cmmf_prec=cmmf_prec,tem_norm=t_tem_norm,comp=comp_to_calculate[k])

        if k == 0:

            q_map = q_map_tem

        j_clus = int(np.floor(theta_x/pix.dx))
        i_clus = int(pix.ny-np.ceil(theta_y/pix.dy))

        q_opt[k] = q_map_tem[i_clus,j_clus]
        y0_est[k] = y_map[i_clus,j_clus]

    return q_opt,y0_est,q_map

def apply_mask_peak_finding(tensor,mask):

    for i in range(0,tensor.shape[2]):

        tensor[:,:,i] *= mask

    return tensor

#Engine for cluster finding

def make_detections(q_tensor,q_th,pix,detection_method="maxima"):

    indices = np.where(q_tensor > q_th)

    coords = np.zeros((len(indices[0]),2),dtype=int)
    coords[:,0] = indices[0]
    coords[:,1] = indices[1]
    theta_coord = np.array(indices[2])

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

            for i in range(0,n_clusters):

                indices = np.where(labels==i)
                q_tensor_select = q_tensor[coords[:,0][indices],coords[:,1][indices],theta_coord[indices]]
                opt_index = np.argmax(q_tensor_select)
                i_opt_vec[i]  = coords[:,0][indices][opt_index]
                j_opt_vec[i] = coords[:,1][indices][opt_index]
                theta_opt_vec[i] = theta_coord[indices][opt_index]
                q_opt = q_tensor_select[opt_index]

            ret = (i_opt_vec,j_opt_vec,theta_opt_vec)

        elif detection_method == "maxima":

            q_tensor_maxima = q_tensor[:,:,:].copy()
            q_tensor_maxima[np.where(q_tensor_maxima < q_th)] = 0.

            maxs_idx = get_maxima(get_maxima_mask(q_tensor_maxima))

            i = np.array(maxs_idx[:,0]).astype(int)
            j = np.array(maxs_idx[:,1]).astype(int)
            theta = np.array(maxs_idx[:,2]).astype(int)

            ret = (i,j,theta)

    return ret

def make_detections2(qmap, ymap, mask_peak_finding, q_th, zmax):
    coords = np.where(np.logical_and(mask_peak_finding, (qmap > q_th)))  # Assume binary mask
    maxs_idx = get_maxima(get_maxima_mask_2d(qmap, coords) & zmax)
    indices = tuple(maxs_idx.astype(int).T) # (idx[:,0], idx[;,1])
    q_opt = qmap[indices]
    y0_est = ymap[indices]
    return q_opt, y0_est, maxs_idx ## arrays: (Ndetections,), (Ndetections,), (Ndetections, 2)

#Mask for peak finding

def get_maxima_mask(r_input):

    r = np.ones((r_input.shape[0]+2,r_input.shape[1]+2,r_input.shape[2]+2)) * -np.inf
    r[1:-1,1:-1,1:-1] = r_input

    xmax=np.argmax([r[2:,1:-1,1:-1],r[:-2,1:-1,1:-1],r[1:-1,1:-1,1:-1]],axis=0) == 2
    ymax=np.argmax([r[1:-1,2:,1:-1], r[1:-1,:-2,1:-1],r[1:-1,1:-1,1:-1]], axis=0) == 2
    zmax=np.argmax([r[1:-1,1:-1, 2:], r[1:-1,1:-1, :-2], r[1:-1,1:-1, 1:-1]], axis=0) == 2

    return xmax&ymax&zmax

def get_maxima_mask_2d(r_input, coords=None):

    r = np.ones((r_input.shape[0]+2, r_input.shape[1]+2)) * -np.inf
    if coords is None:
        r[1:-1, 1:-1] = r_input
    else:
        r[1:-1, 1:-1] = 0
        r[1:-1, 1:-1][coords] = r_input[coords]

    xmax = np.argmax([r[2:, 1:-1], r[:-2, 1:-1], r[1:-1, 1:-1]], axis=0) == 2
    ymax = np.argmax([r[1:-1, 2:], r[1:-1, :-2], r[1:-1, 1:-1]], axis=0) == 2

    return xmax&ymax

#Peak finding

def get_maxima(max_mask):

    maxima = np.argwhere(max_mask == True)

    return maxima

#Get SNR map

def get_mmf_q_map(tmap,tem,inv_cov,pix,theta_misc_template=[0.,0.],
mmf_type="standard",cmmf_prec=None,comp=0,tem_norm=None):
    n_freqs = tmap.shape[2]
    y_map = np.zeros((pix.nx,pix.ny), dtype=tmap.dtype)
    #norm_map = np.zeros((pix.nx,pix.ny))

    tem_fft = maps.get_fft_f(tem,pix)

    filter_fft = get_tem_conv_fft(pix,tem_fft,inv_cov,mmf_type=mmf_type,
    cmmf_prec=cmmf_prec,comp=comp)
    del tem_fft

    filter = np.asarray(maps.get_ifft_f(filter_fft,pix).real, dtype=tmap.dtype)
    del filter_fft

    tem[:] = maps.get_tmap_times_fvec(tem,cmmf_prec.a_matrix[:,comp]).real #new line

    if tem_norm is None:

        tem_norm = tem

    else:

        tem_norm = maps.get_tmap_times_fvec(tem_norm,cmmf_prec.a_matrix[:,comp]).real #new line
        del tem


    #mask = get_apodised_mask(pix,np.ones((pix.nx,pix.nx)),apotype="Smooth",aposcale=0.2)

    for i in range(n_freqs):

        y = sg.fftconvolve(tmap[:,:,i],filter[:,:,i],mode='same')*pix.dx*pix.dy

        #y = maps.fftconvolve(tmap[:,:,i] * xmask, filter[:,:,i]) * pix.dx * pix.dy ## Check you use the right function

        #norm = sg.fftconvolve(tem_norm[:,:,i],filter[:,:,i],mode='same')*pix.dx*pix.dy

        y_map += y

    y_map = np.asarray(y_map, dtype=tmap.dtype)
    del y
        #norm_map += norm

    norm = np.sum(tem_norm*filter)*pix.dx*pix.dy
    del tem_norm, filter
    #norm = np.max(norm_map)

    y_map = y_map/norm

    std = 1./np.sqrt(norm)
    q_map = y_map/std

    return q_map,y_map,std

def get_tem_conv_fft(pix_tem,tem_fft,inv_cov,mmf_type="standard",cmmf_prec=None,comp=0):

    tem_fft = maps.reshape_ell_matrix(tem_fft, inv_cov.shape[:2]) # Match to cut inv_cov

    if mmf_type == "standard":

        tem_fft = maps.get_tmap_times_fvec(tem_fft,cmmf_prec.a_matrix[:,comp]) #new line
        filter_fft = utils.get_inv_cov_conjugate(tem_fft,inv_cov)

    elif mmf_type == "spectrally_constrained": #doesn't admit changing compoenent order

        if cmmf_prec.cmmf_type == "one_dep":

            n_freq = tem_fft.shape[2]
            a_dot_b =  maps.get_tmap_from_map(cmmf_prec.a_dot_b,n_freq)
            b_dot_b = maps.get_tmap_from_map(cmmf_prec.b_dot_b,n_freq)
            b = cmmf_prec.a_matrix[:,1] #cmmf_prec.sed_b
            a = cmmf_prec.a_matrix[:,0] #cmmf_prec.sed_a

            tem_fft_b = maps.get_tmap_times_fvec(tem_fft,b)
            tem_fft = maps.get_tmap_times_fvec(tem_fft,a)

            filter_fft = utils.get_inv_cov_conjugate(tem_fft,inv_cov) - a_dot_b/b_dot_b*utils.get_inv_cov_conjugate(tem_fft_b,inv_cov)

        elif cmmf_prec.cmmf_type == "general":

            filter_fft_0 = cmmf_prec.w_norm_list[comp]
            filter_fft = filter_fft_0*tem_fft

    filter_fft[np.isnan(filter_fft)] = 0.
    filter_fft = maps.reshape_ell_matrix(filter_fft, (pix_tem.nx, pix_tem.ny)) # Expand back to expected shape
    return filter_fft

#Class to compute weights for spectrally constrained MMF

class scmmf_precomputation:

    #sed's in muK

    def __init__(self,pix=None,freqs=None,inv_cov=None,lrange=[0,1000000],beam_type="gaussian",exp=None,
    mask=None,cmmf_type="one_dep",a_matrix=None,n_dep=1,comp_to_calculate=[0],mmf_type="standard"):

        self.cmmf_type = cmmf_type
        self.a_matrix = a_matrix
        if mask is None:

            mask = np.ones((pix.nx,pix.ny,len(freqs)))

        if self.cmmf_type == "one_dep" and mmf_type == "spectrally_constrained":
            print("ONE_DEP")
            self.sed_a = a_matrix[:,0][freqs]
            self.sed_b = a_matrix[:,1][freqs]

            a_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
            a_map_fft = maps.get_tmap_times_fvec(a_map_fft,self.sed_a)

            a_map_fft = maps.get_map_convolved_fft(a_map_fft,pix,freqs,beam_type,mask,lrange,exp)
            # TODO: More efficient to do next line earlier, but requires modifications to the functions in maps
            a_map_fft = maps.reshape_ell_matrix(a_map_fft, inv_cov.shape[:2]) # Cut to correct lrange

            b_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
            b_map_fft = maps.get_tmap_times_fvec(b_map_fft,self.sed_b)

            b_map_fft = maps.get_map_convolved_fft(b_map_fft,pix,freqs,beam_type,mask,lrange,exp)
            b_map_fft = maps.reshape_ell_matrix(b_map_fft, inv_cov.shape[:2])

            self.a_dot_b = utils.get_inv_cov_dot(np.conjugate(a_map_fft),inv_cov,b_map_fft)
            self.b_dot_b = utils.get_inv_cov_dot(np.conjugate(b_map_fft),inv_cov,b_map_fft)
            del a_map_fft
            del b_map_fft

        elif self.cmmf_type == "general" and mmf_type == "spectrally_constrained":
            print("GENERAL")
            #a_matrix is f x c matrix (f number of frequencies, c number of components, 1st SZ, rest to deproject)

            a_matrix_fft = np.zeros((pix.nx,pix.nx,len(freqs),self.a_matrix.shape[1]))

            for i in range(0,a_matrix.shape[1]):

                a_matrix_fft_i = np.ones((pix.nx,pix.nx,len(freqs)))
                a_matrix_fft_i = maps.get_tmap_times_fvec(a_matrix_fft_i,self.a_matrix[freqs,i])
                a_matrix_fft[:,:,:,i] = maps.get_map_convolved_fft(a_matrix_fft_i,pix,freqs,beam_type,mask,lrange,exp)
                #a_matrix_fft[:,:,:,i] = a_matrix_fft_i
            del a_matrix_fft_i

            a_matrix_fft = maps.reshape_ell_matrix(a_matrix_fft, inv_cov.shape[:2])
            #self.a_matrix_fft = a_matrix_fft

            pre = np.einsum('abcd,abde->abce',inv_cov,a_matrix_fft)
            dot = utils.invert_cov(np.einsum('abdc,abde->abce',a_matrix_fft,pre))
            del a_matrix_fft
            W = np.einsum('abcd,abed->abce',dot,pre)
            cov = utils.invert_cov(inv_cov)

            # self.w_list = []
            # self.sigma_y_list = []
            self.w_norm_list = []

            for i in range(0,len(comp_to_calculate)):

                # e_map = np.zeros((W.shape[0],W.shape[1],self.a_matrix.shape[1]))
                # e_map[:,:,comp_to_calculate[i]] = 1.

                # w = np.einsum('ija,ijab->ijb',e_map,W)
                w = W[:,:,comp_to_calculate[i]]

                sigma_y = utils.get_inv_cov_dot(np.conjugate(w),cov,w)

                w_norm = np.zeros(w.shape)

                for i in range(0,len(freqs)):

                    w_norm[:,:,i] = w[:,:,i]/sigma_y

                # self.w_list.append(w)
                # self.sigma_y_list.append(sigma_y)
                self.w_norm_list.append(w_norm)

        elif mmf_type == "standard":

            r = 1.

        else:

            if mask is None:

                mask = np.ones((pix.nx,pix.ny,len(freqs)))

            if self.cmmf_type == "one_dep" and mmf_type == "spectrally_constrained":

                self.sed_a = a_matrix[:,0][freqs]
                self.sed_b = a_matrix[:,1][freqs]

                a_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
                a_map_fft = maps.get_tmap_times_fvec(a_map_fft,self.sed_a)

                a_map_fft = maps.get_map_convolved_fft(a_map_fft,pix,freqs,beam_type,mask,lrange,exp)

                b_map_fft = np.ones((pix.nx,pix.nx,len(freqs)))
                b_map_fft = maps.get_tmap_times_fvec(b_map_fft,self.sed_b)

                b_map_fft = maps.get_map_convolved_fft(b_map_fft,pix,freqs,beam_type,mask,lrange,exp)

                self.a_dot_b = utils.get_inv_cov_dot(np.conjugate(a_map_fft),inv_cov,b_map_fft)
                self.b_dot_b = utils.get_inv_cov_dot(np.conjugate(b_map_fft),inv_cov,b_map_fft)

            elif self.cmmf_type == "general" and mmf_type == "spectrally_constrained":

                #a_matrix is f x c matrix (f number of frequencies, c number of components, 1st SZ, rest to deproject)

                a_matrix_fft = np.zeros((pix.nx,pix.nx,len(freqs),self.a_matrix.shape[1]))

                for i in range(0,a_matrix.shape[1]):

                    a_matrix_fft_i = np.ones((pix.nx,pix.nx,len(freqs)))
                    a_matrix_fft_i = maps.get_tmap_times_fvec(a_matrix_fft_i,self.a_matrix[freqs,i])
                    a_matrix_fft[:,:,:,i] = maps.get_map_convolved_fft(a_matrix_fft_i,pix,freqs,beam_type,mask,lrange,exp)
                    #a_matrix_fft[:,:,:,i] = a_matrix_fft_i

                self.a_matrix_fft = a_matrix_fft

                pre = np.einsum('abcd,abde->abce',inv_cov,a_matrix_fft)
                dot = utils.invert_cov(np.einsum('abdc,abde->abce',a_matrix_fft,pre))
                self.W = np.einsum('abcd,abed->abce',dot,pre)
                cov = utils.invert_cov(inv_cov)

                self.w_list = []
                self.sigma_y_list = []
                self.w_norm_list = []

                for i in range(0,len(comp_to_calculate)):

                    e_map = np.zeros((pix.nx,pix.nx,self.a_matrix.shape[1]))
                    e_map[:,:,comp_to_calculate[i]] = 1.

                    w = np.einsum('ija,ijab->ijb',e_map,self.W)

                    sigma_y = utils.get_inv_cov_dot(np.conjugate(w),cov,w)

                    w_norm = np.zeros(w.shape)

                    for i in range(0,len(freqs)):

                        w_norm[:,:,i] = w[:,:,i]/sigma_y

                    self.w_list.append(w)
                    self.sigma_y_list.append(sigma_y)
                    self.w_norm_list.append(w_norm)



def get_a_matrix_cib(params_szifi,params_model,data_file):

    cib = sed.cib_model(params_model=params_model)
    cib_sed = cib.get_sed_muK_experiment(experiment=data_file["experiment"],bandpass=params_szifi["integrate_bandpass"])
    cib.get_sed_first_moments_experiment(experiment=data_file["experiment"],
    bandpass=params_szifi["integrate_bandpass"],moment_parameters=params_szifi["deproject_cib"])

    freqs = params_szifi["freqs"]

    a_matrix = np.zeros((len(freqs),len(params_szifi["deproject_cib"])))

    if params_szifi["deproject_cib"] == ["cib"]:

        a_matrix = np.zeros((len(freqs),2))
        a_matrix[:,0] = data_file["experiment"].tsz_sed[freqs]
        a_matrix[:,1] = cib_sed[freqs]

    elif params_szifi["deproject_cib"] == ["cib","betaT"]:

        a_matrix = np.zeros((len(freqs),3))
        a_matrix[:,0] = data_file["experiment"].tsz_sed[freqs]
        a_matrix[:,1] = cib_sed[freqs]
        a_matrix[:,2] = cib.moments["betaT"][freqs]

    elif params_szifi["deproject_cib"] == ["cib","beta"]:

        a_matrix = np.zeros((len(freqs),3))
        a_matrix[:,0] = data_file["experiment"].tsz_sed[freqs]
        a_matrix[:,1] = cib_sed[freqs]
        a_matrix[:,2] = cib.moments["beta"][freqs]

    elif params_szifi["deproject_cib"] == ["cib","betaT","beta"]:

        a_matrix = np.zeros((len(freqs),3))
        a_matrix[:,0] = data_file["experiment"].tsz_sed[freqs]
        a_matrix[:,1] = cib_sed[freqs]
        a_matrix[:,2] = cib.moments["betaT"][freqs]
        a_matrix[:,3] = cib.moments["beta"][freqs]

    params_szifi["a_matrix"] = a_matrix
