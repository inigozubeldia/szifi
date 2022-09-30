import numpy as np
import pylab as pl
import scipy.special as sp
import healpy as hp
import sphere
import maps
import scipy.integrate as integrate
from sklearn.cluster import DBSCAN
from random import randint
from random import randrange
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import websky
import model
from astropy.cosmology import Planck15
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import copy
import pickle

#Unlike in the rest of the code, theta_x is actual x angular coordinate  (id for theta_y)
# w.r.t to the origin (lower left) of the field.


#standard keys used throughout:

#q_opt
#y0
#theta_500
#theta_x
#theta_y
#pixel_ids
#lat
#lon
#m_500
#z

class cluster_catalogue:

    def __init__(self):

        self.catalogue = {}
        self.initialise()

    def append(self,new_catalogue,append_keys="old"): #note: catalogues must have the same keys. Iteration over the keys of the class catalogue

        if append_keys == "old":

            keys = self.catalogue.keys()

        elif append_keys == "new":

            keys = new_catalogue.catalogue.keys()

        for key in keys:

            if key in self.catalogue:

                self.catalogue[key] = np.append(self.catalogue[key],new_catalogue.catalogue[key])

            else:

                self.catalogue[key] = new_catalogue.catalogue[key]



    def select_indices(self,indices):

        for key in self.catalogue.keys():

            if len(self.catalogue[key]) > 0:

                self.catalogue[key] = self.catalogue[key][indices]


    def get_lonlat(self,n_side,i,pix):

        lon,lat = sphere.get_lonlat(i,self.catalogue["theta_x"]-pix.nx*pix.dx*0.5,self.catalogue["theta_y"]-pix.nx*pix.dx*0.5,n_side)
        self.catalogue["lon"] = lon
        self.catalogue["lat"] = lat

    def select_tile(self,i,pix,type="field",nside=8): #it works

        theta_x,theta_y = sphere.get_xy(i,self.catalogue["lon"],self.catalogue["lat"],nside)

        if type == "field":

            lx = pix.nx*pix.dx
            indices = np.where((theta_x < lx*0.5) & (theta_x > -lx*0.5) & (theta_y < lx*0.5) & (theta_y > -lx*0.5))
            theta_x = theta_x + lx*0.5
            theta_y = theta_y + lx*0.5

        elif type == "tile":

            pix_vec = hp.pixelfunc.ang2pix(nside,self.lon,self.lat,lonlat=True)
            indices = np.where((pix_vec == i))

        catalogue = get_catalogue_indices(self,indices)

        if type == "field":

            catalogue.catalogue["theta_x"] = theta_x[indices]
            catalogue.catalogue["theta_y"] = theta_y[indices]

        return catalogue

    def initialise(self):

        self.catalogue["q_opt"] = np.empty(0)
        self.catalogue["y0"] = np.empty(0)
        self.catalogue["theta_500"] = np.empty(0)
        self.catalogue["theta_x"] = np.empty(0)
        self.catalogue["theta_y"] = np.empty(0)
        self.catalogue["lon"] = np.empty(0)
        self.catalogue["lat"] = np.empty(0)
        self.catalogue["m_500"] = np.empty(0)
        self.catalogue["z"] = np.empty(0)
        self.catalogue["pixel_ids"] = np.empty(0)

    def get_n_clusters(self):

        return len(self.catalogue["q_opt"][np.where(self.catalogue["q_opt"] != -1.)[0]])

def merge_detections(catalogue,radius_arcmin=10.,return_merge_flag=False,mode="closest"):

    catalogue = get_catalogue_indices(catalogue,np.where(catalogue.catalogue["q_opt"] != -1.)[0])
    n_clusters = len(catalogue.catalogue["q_opt"])
    catalogue_merged = cluster_catalogue()
    catalogue_compare = catalogue

    if mode == "closest":

        i = 0

        indices_subtract = np.arange(len(catalogue_compare.catalogue["q_opt"]))
        merge_flags = np.zeros(0)

        while len(indices_subtract) > 0:

            catalogue_compare_new = cluster_catalogue()

            for key in catalogue_compare.catalogue.keys():

                if len(catalogue_compare.catalogue[key]) > 0.:

                    catalogue_compare_new.catalogue[key] = catalogue_compare.catalogue[key][indices_subtract]

            catalogue_compare = catalogue_compare_new

            n_clusters = len(catalogue_compare.catalogue["q_opt"])
            dist = sphere.get_angdist(catalogue_compare.catalogue["lon"][0]*np.ones(n_clusters),catalogue_compare.catalogue["lon"],
            catalogue_compare.catalogue["lat"][0]*np.ones(n_clusters),catalogue_compare.catalogue["lat"])

            indices = np.where(dist < radius_arcmin/180./60.*np.pi)[0]
            merge_flags = np.append(merge_flags,len(indices))

            index_max = indices[np.argmax(catalogue_compare.catalogue["q_opt"][indices])]

            catalogue_new = cluster_catalogue()

            for key in catalogue_compare.catalogue.keys():

                if len(catalogue_compare.catalogue[key]) > 0.:

                    catalogue_new.catalogue[key] = np.array([catalogue_compare.catalogue[key][index_max]])

            catalogue_merged.append(catalogue_new,append_keys="new")

            indices_subtract = np.setdiff1d(np.arange(len(catalogue_compare.catalogue["q_opt"])),indices)

            i += 1

        if return_merge_flag == True:

            ret = (catalogue_merged,merge_flags)

        else:

            ret = catalogue_merged

    elif mode == "fof":

        coords = np.zeros((n_clusters,2))
        coords[:,0] = catalogue.catalogue["lon"]
        coords[:,1] = catalogue.catalogue["lat"]

        clust = AgglomerativeClustering(distance_threshold=radius_arcmin,
        linkage="single",n_clusters=None,compute_full_tree=True,
        affinity=get_affinity_spherical)
        clust.fit(coords)
        labels = clust.labels_
        n_clusters_merged = np.max(labels)+1

        for i in range(0,n_clusters_merged):

            indices = np.where(labels==i)
            q_max = np.max(catalogue.catalogue["q_opt"][indices])
            index_max = np.where(catalogue.catalogue["q_opt"] == q_max)

            catalogue_new = cluster_catalogue()

            for key in catalogue_compare.catalogue.keys():

                if len(catalogue.catalogue[key]) > 0.:

                    catalogue_new.catalogue[key] = np.array([catalogue.catalogue[key][index_max]])

            catalogue_merged.append(catalogue_new,append_keys="new")

        ret = catalogue_merged

    return ret

def get_affinity_spherical(X):

    return pairwise_distances(X,X,metric=get_distance_sphere_lonlat)

def get_distance_sphere_lonlat(coords1,coords2):

    lon1 = coords1[0]
    lat1 = coords1[1]
    lon2 = coords2[0]
    lat2 = coords2[1]

    distance = hp.rotator.angdist((lon1,lat1),(lon2,lat2),lonlat=True)*180.*60./np.pi

    return distance

#Cross-match cluster catalogues. catalogue_ref is the reference catalogue, whose ordering is
#kept.

def identify_clusters(catalogue_1,catalogue_2,id_radius_arcmin=15.,lonlat=False,
mode="close+highest_q",sort=True,unique=False):

    catalogue_1_new = copy.deepcopy(catalogue_1)
    catalogue_2_new = cluster_catalogue()
    indices_catalogue_2 = np.zeros(catalogue_2.get_n_clusters())

    if sort == True:

        indices_sorted = np.flip(np.argsort(catalogue_1_new.catalogue["q_opt"]))
        catalogue_1_new = get_catalogue_indices(catalogue_1_new,indices_sorted)
        catalogue_1 = get_catalogue_indices(catalogue_1,indices_sorted)

    for i in range(0,catalogue_1.get_n_clusters()):

        if lonlat == False:

            dist = np.sqrt((catalogue_1.catalogue["theta_x"][i]-catalogue_2.catalogue["theta_x"])**2+(catalogue_1.catalogue["theta_y"][i]-catalogue_2.catalogue["theta_y"])**2)

        elif lonlat == True:

            dist = sphere.get_angdist(catalogue_1.catalogue["lon"][i],catalogue_2.catalogue["lon"],catalogue_1.catalogue["lat"][i],catalogue_2.catalogue["lat"])

        if unique == False:

            indices_closest = np.where(dist < id_radius_arcmin/180./60.*np.pi)[0]

        elif unique == True:

            indices_closest = np.where((dist < id_radius_arcmin/180./60.*np.pi) & (indices_catalogue_2 == 0.))[0]

        if len(indices_closest) != 0:

            if mode == "close+highest_q":  #in this mode, catalogue_1 should be the observed catalogue

                index_max = indices_closest[np.argmax(catalogue_2.catalogue["q_opt"][indices_closest])]

            elif mode == "closest":

                index_max = indices_closest[np.argmin(dist[indices_closest])]

            cat_temp = cluster_catalogue()

            for key in catalogue_2.catalogue.keys():

                if len(catalogue_2.catalogue[key]) > 0.:

                    cat_temp.catalogue[key] = catalogue_2.catalogue[key][index_max]

            catalogue_2_new.append(cat_temp,append_keys="new")

            indices_catalogue_2[index_max] = 1.

        else:

            cat_temp = cluster_catalogue()

            for key in catalogue_2.catalogue.keys():

                if len(catalogue_2.catalogue[key]) > 0.:

                    cat_temp.catalogue[key] = np.array([-1])

            catalogue_2_new.append(cat_temp,append_keys="new")

    indices = np.where(indices_catalogue_2==0.)[0]

    for i in indices:

        cat_temp_1 = cluster_catalogue()
        cat_temp_2 = cluster_catalogue()

        for key in catalogue_1.catalogue.keys():

            if len(catalogue_1.catalogue[key]) > 0.:

                cat_temp_1.catalogue[key] = np.array([-1])

        for key in catalogue_2.catalogue.keys():

            if len(catalogue_2.catalogue[key]) > 0.:

                cat_temp_2.catalogue[key] = np.array([catalogue_2.catalogue[key][i]])

        catalogue_1_new.append(cat_temp_1,append_keys="new")
        catalogue_2_new.append(cat_temp_2,append_keys = "new")

    return (catalogue_1_new,catalogue_2_new)


def apply_mask_select(cat,mask_select,pix):

    i,j = maps.get_ij_from_theta(cat.catalogue["theta_x"],cat.catalogue["theta_y"],pix)
    mask_values = mask_select[i,j]
    indices = np.where(mask_values != 0.)[0]

    return get_catalogue_indices(cat,indices)

def apply_mask_select_fullsky(cat,mask_select):

    nside = hp.npix2nside(len(mask_select))
    pixs = hp.ang2pix(nside,cat.catalogue["lon"],cat.catalogue["lat"],lonlat=True)
    mask_values = mask_select[pixs]
    indices = np.where(mask_values != 0.)[0]

    return get_catalogue_indices(cat,indices)

def get_catalogue_sky_selected(cat,theta_x_range,theta_y_range=None):

    if theta_y_range is None:

        theta_y_range = theta_x_range

    indices = np.where((cat.catalogue["theta_x"] >= theta_x_range[0]) & (cat.catalogue["theta_x"] < theta_x_range[1])
    & (cat.catalogue["theta_y"] >= theta_y_range[0]) & (cat.catalogue["theta_y"] < theta_y_range[1]))[0]

    return get_catalogue_indices(cat,indices)

def get_catalogue_q_th(cat,q_th):

    indices = np.where(cat.catalogue["q_opt"] >= q_th)

    return get_catalogue_indices(cat,indices)

def get_catalogue_indices(cat_old,indices):

    cat_new = cluster_catalogue()

    for key in cat_old.catalogue.keys():

        if len(cat_old.catalogue[key]) > 0.:

            cat_new.catalogue[key] = cat_old.catalogue[key][indices]

    return cat_new

def remove_catalogue_indices(cat,indices_to_remove):

    n = len(cat.catalogue["q_opt"])
    indices = np.delete(np.arange(0,n),indices_to_remove)

    return get_catalogue_indices(cat,indices)

def remove_catalogue_tiles(cat,tiles_to_remove):

    tiles_to_remove = tiles_to_remove.astype("int")
    indices_to_remove = np.array([],dtype="int")

    for i in range(0,len(tiles_to_remove)):

        indices_to_remove = np.append(indices_to_remove,np.where((cat.catalogue["pixel_ids"] == tiles_to_remove[i]))[0])

    return remove_catalogue_indices(cat,indices_to_remove),indices_to_remove

def get_true_positives(catalogue_1,catalogue_2):

    q_1 = catalogue_1.catalogue["q_opt"]
    q_2 = catalogue_2.catalogue["q_opt"]
    indices = np.where((q_1 != -1.) & (q_2 != -1.))

    catalogue_1.select_indices(indices)
    catalogue_2.select_indices(indices)

    return (catalogue_1,catalogue_2)


class catalogue_comparer:

    def __init__(self,catalogue_true,catalogue_obs):

        self.catalogue_true = catalogue_true
        self.catalogue_obs = catalogue_obs

    def get_true_positives(self,q_th_obs=0.,q_th_true=0.,q_max_obs=np.inf):

        indices = np.where((self.catalogue_true.catalogue["q_opt"] != -1.) & (self.catalogue_obs.catalogue["q_opt"] != -1.) &
        (self.catalogue_true.catalogue["q_opt"] >= q_th_true) & (self.catalogue_obs.catalogue["q_opt"] >= q_th_obs) &
        (self.catalogue_obs.catalogue["q_opt"] < q_max_obs))[0]

        return (get_catalogue_indices(self.catalogue_true,indices),get_catalogue_indices(self.catalogue_obs,indices))

    def get_n_true_positives(self,q_th_obs=0.,q_th_true=0.,q_max_obs=np.inf):

        return len(self.get_true_positives(q_th_obs=q_th_obs,q_th_true=q_th_true,q_max_obs=q_max_obs)[0].catalogue["q_opt"])

    def get_false_positives(self,q_th_obs=0.,q_th_true=0.):

        indices = np.where((self.catalogue_true.catalogue["q_opt"] == -1.) & (self.catalogue_obs.catalogue["q_opt"] != -1.) & (self.catalogue_obs.catalogue["q_opt"] >= q_th_obs))[0]

        return (get_catalogue_indices(self.catalogue_true,indices),get_catalogue_indices(self.catalogue_obs,indices))

    def get_n_false_positives(self,q_th_obs=0.,q_th_true=0.):

        return len(self.get_false_positives(q_th_obs=q_th_obs,q_th_true=q_th_true)[0].catalogue["q_opt"])

    def get_undetected(self,q_th_obs=0.,q_th_true=0.):

        indices = np.where((self.catalogue_true.catalogue["q_opt"] != -1.) & (self.catalogue_obs.catalogue["q_opt"] == -1. ) & (self.catalogue_true.catalogue["q_opt"] >= q_th_true))[0]

        return (get_catalogue_indices(self.catalogue_true,indices),get_catalogue_indices(self.catalogue_obs,indices))

    def get_n_undetected(self,q_th_obs=0.,q_th_true=0.):

        return len(self.get_undetected(q_th_obs=q_th_obs,q_th_true=q_th_true)[0].catalogue["q_opt"])

    def get_n_detected(self,q_th_obs=0.,q_th_true=0.,q_max_obs=np.inf):

        indices = np.where((self.catalogue_obs.catalogue["q_opt"] >= q_th_obs) & (self.catalogue_obs.catalogue["q_opt"] < q_max_obs))[0]

        return len(indices)

    def get_n_true(self,q_th_true=0.):

        indices = np.where(self.catalogue_true.catalogue["q_opt"] >= q_th_true)[0]

        return len(indices)

    def get_purity(self,q_th_obs=0.,q_max_obs=np.inf):

        n_detected = self.get_n_detected(q_th_obs=q_th_obs,q_max_obs=q_max_obs)

        if n_detected == 0:

            ret = -1.

        else:

            ret = self.get_n_true_positives(q_th_obs=q_th_obs,q_max_obs=q_max_obs)/n_detected

        #print(q_th_obs,self.get_n_true_positives(q_th_obs=q_th_obs,q_max_obs=q_max_obs),n_detected)

        return ret

    def get_cumulative_completeness(self,q_th_true=0.):

        n_true = self.get_n_true(q_th_true=q_th_true)

        if n_true == 0:


            ret = -1.

        else:

            ret = self.get_n_true_positives(q_th_true=q_th_true)/n_true

        return ret

    def get_completeness_bin(self,q_true_min,q_true_max,q_th=4.5):

        indices_true_pos = np.where((self.catalogue_true.catalogue["q_opt"] != -1.) & (self.catalogue_obs.catalogue["q_opt"] != -1.)
        & (self.catalogue_true.catalogue["q_opt"] >= q_true_min) & (self.catalogue_true.catalogue["q_opt"] < q_true_max) & (self.catalogue_obs.catalogue["q_opt"] >= q_th))[0]
        indices_true = np.where((self.catalogue_true.catalogue["q_opt"] >= q_true_min) & (self.catalogue_true.catalogue["q_opt"] < q_true_max))[0]

        n_true_pos = len(indices_true_pos)
        n_true = len(indices_true)

        if n_true == 0:

            ret = (0.,0.)

        else:

            ret =  (n_true_pos/n_true,np.sqrt(n_true_pos)/n_true)

        return ret

    def get_completeness(self,bins,q_th=4.5):

        completeness = np.zeros(len(bins)-1)
        bins_centres = np.zeros(len(bins)-1)
        completeness_error = np.zeros((len(bins)-1))

        for i in range(0,len(completeness)):

            completeness[i],completeness_error[i] = self.get_completeness_bin(bins[i],bins[i+1],q_th=q_th)
            bins_centres[i] = 0.5*(bins[i+1]+bins[i])

        return bins_centres,completeness,completeness_error

    def get_theta_std_bin(self,q_min,q_max): #for true positives

        cat_true,cat_obs = self.get_true_positives(q_th_obs=q_min,q_max_obs=q_max)
        distances = sphere.get_angdist(cat_true.catalogue["lon"],cat_obs.catalogue["lon"],cat_true.catalogue["lat"],cat_obs.catalogue["lat"])/np.pi*60.*180.
        theta_std = np.mean(distances)

        return theta_std

    def get_theta_std(self,bins):

        theta_std = np.zeros(len(bins)-1)
        bins_centres = np.zeros(len(bins)-1)

        for i in range(0,len(theta_std)):

            theta_std[i] = self.get_theta_std_bin(bins[i],bins[i+1])
            bins_centres[i] = 0.5*(bins[i+1]+bins[i])

        return bins_centres,theta_std

    def get_theta_500_bias(self,bins):

        theta_500_bias = np.zeros(len(bins)-1)
        theta_500_bias_std = np.zeros(len(bins)-1)
        bins_centres = np.zeros(len(bins)-1)

        for i in range(0,len(bins_centres)):

            cat_true,cat_obs = self.get_true_positives(q_th_obs=bins[i],q_max_obs=bins[i+1])
            ratio = cat_obs.catalogue["theta_500"]/cat_true.catalogue["theta_500"]
            theta_500_bias[i] = np.mean(ratio)
            theta_500_bias_std[i] = np.std(ratio)/np.sqrt(float(len(ratio)))
            bins_centres[i] = 0.5*(bins[i+1]+bins[i])

        return bins_centres,theta_500_bias,theta_500_bias_std

def get_completeness_err(bins_edges,catalogue_true,catalogue_obs,q_th=4.5,n_boots=100):

    completeness_boots = np.zeros((n_boots,len(bins_edges)-1))

    for i in range(0,n_boots):

        n_clusters = len(catalogue_true.catalogue["q_opt"])
        indices = np.random.randint(0,n_clusters,size=n_clusters)
        cat_true_boots = get_catalogue_indices(catalogue_true,indices)
        cat_obs_boots = get_catalogue_indices(catalogue_obs,indices)
        comparison = catalogue_comparer(cat_true_boots,cat_obs_boots)
        bins,completeness,completeness_err = comparison.get_completeness(bins_edges,q_th=q_th)
        completeness_boots[i,:] = completeness

    completeness_err = np.std(completeness_boots,axis=0)

    return bins,completeness_err


def get_erf_completeness(q_true,q_th=4.5,opt_bias=True,sigma=1.):

    if opt_bias == True:

        bias = 3.

    elif opt_bias == False:

        bias = 0.

    elif opt_bias == 2:

        bias = 2.

    q_true = np.sqrt(q_true**2 + bias)

    return (1.-sp.erf((q_th-q_true)/(np.sqrt(2.)*sigma)))*0.5

def get_erf_completeness_binned(bins_edges,q_th=4.5,opt_bias=True):

    binned_completeness = np.zeros(len(bins_edges)-1)
    bins_centres = np.zeros(len(bins_edges)-1)

    for i in range(0,len(binned_completeness)):

        def int_f(q):

            return get_erf_completeness(q,q_th=q_th,opt_bias=opt_bias)

        binned_completeness[i] = integrate.quad(int_f,bins_edges[i],bins_edges[i+1])[0]/(bins_edges[i+1]-bins_edges[i])
        bins_centres[i] = (bins_edges[i] + bins_edges[i+1])*0.5

    return bins_centres,binned_completeness


class results_detection_old:

    def __init__(self,results_grid=cluster_catalogue(),results_refined=cluster_catalogue(),
    results_grid_noit=cluster_catalogue(),results_refined_noit=cluster_catalogue(),results_true=cluster_catalogue()):

        self.results_grid = results_grid
        self.results_refined = results_refined
        self.results_grid_noit = results_grid_noit
        self.results_refined_noit = results_refined_noit
        self.results_true = results_true
        self.sigma_vec = None
        self.sigma_vec_noit = None
        self.theta_500_sigma = None
        self.f_sky = None
        self.cib_id = None
        self.info = None

    def append(self,new_results):

        self.results_grid = self.results_grid.append(new_results.results_grid,append_keys="new")
        self.results_refined = self.results_refined.append(new_results.results_refined,append_keys="new")
        self.results_grid_noit = self.results_grid_noit.append(new_results.results_grid_noit,append_keys="new")
        self.results_refined_noit = self.results_refined_noit.append(new_results.results_refined_noit,append_keys="new")
        self.results_true = self.results_true.append(new_results.results_true,append_keys="new")

    def set_pixel_ids(self,pixel_ids):

        self.results_grid.catalogue["pixel_ids"] = pixel_ids
        self.results_refined.catalogue["pixel_ids"] = pixel_ids
        self.results_grid_noit.catalogue["pixel_ids"] = pixel_ids
        self.results_refined_noit.catalogue["pixel_ids"] = pixel_ids
        self.results_true.catalogue["pixel_ids"] = pixel_ids

    def get_lonlat(self,i,pix,n_side=8):

        self.results_grid.get_lonlat(n_side,i,pix)
        self.results_refined.get_lonlat(n_side,i,pix)
        self.results_grid_noit.get_lonlat(n_side,i,pix)
        self.results_refined_noit.get_lonlat(n_side,i,pix)
        self.results_true.get_lonlat(n_side,i,pix)

    def initialise(self):

        for catalogue in [self.results_grid,self.results_refined,self.results_grid_noit,self.results_refined_noit]:

            catalogue.initialise()

class results_detection:

    def __init__(self):

        self.catalogues = {}
        self.sigma_vec = None
        self.sigma_vec_noit = None
        self.theta_500_sigma = None
        self.f_sky = None
        self.info = None
    """

    def append(self,new_results,append_keys="new"):

        if append_keys == "old":

            keys = self.catalogues.keys()

        elif append_keys == "new":

            keys = new_results.catalogues.keys()

        for key in keys:

            if key in self.catalogues:

                self.catalogues[key] = self.catalogues[key].append(new_results.catalogues[key],append_keys=append_keys)

            else:

                self.catalogues[key] = new_catalogue.catalogues[key]
    """

    def set_pixel_ids(self,pixel_ids):

        for key in self.catalogues.keys():

            self.catalogues[key].catalogue["pixel_ids"] = pixel_ids

    def get_lonlat(self,i,pix,n_side=8):

        for key in self.catalogues.keys():

            self.catalogues[key].get_lonlat(n_side,i,pix)

    """
    def initialise(self):

        for key in self.catalogues.keys():

            self.catalogues[key].initialise()
    """

def apply_q_cut(cat,q_th):

    indices = np.where(cat.catalogue["q_opt"] > q_th)[0]

    return get_catalogue_indices(cat,indices)


def get_random_pairs(i_min,i_max):

    random_indices = np.random.choice(np.arange(i_min,i_max),size=(i_max-i_min),replace=False)
    random_pairs = np.zeros(((i_max-i_min)//2,2))
    random_pairs[:,0] = random_indices[0:(i_max-i_min)//2]
    random_pairs[:,1] = random_indices[(i_max-i_min)//2:i_max-i_min]
    random_pairs = random_pairs.astype(int)

    return random_pairs

def get_random_location_tile(mask,pix,n_points):

    theta_x_vec = np.zeros(n_points)
    theta_y_vec = np.zeros(n_points)
    nx,ny = mask.shape

    for i in range(0,n_points):

        ix = 0
        jx = 0

        while mask[ix,jx] == 0.:

            ix = randrange(nx)
            jx = randrange(ny)

        theta_x,theta_y = maps.get_theta_from_ij(ix,jx,pix)
        theta_x_vec[i] = theta_x
        theta_y_vec[i] = theta_y

    return theta_x_vec,theta_y_vec

#catalogue_true mustn't have -1's

def get_q_true_scal_rel(catalogue_true,cosmo="websky",type="arnaud",get_y=False):

    if cosmo == "websky":

        cosmology = websky.cosmology_websky().cosmology

    if cosmo == "Planck15":

        cosmology = Planck15

    n_clus = len(catalogue_true.catalogue["q_opt"])
    y0_scal_rel = np.zeros(n_clus)

    for i in range(n_clus):

        cluster = model.gnfw_tsz(catalogue_true.catalogue["m_500"][i]/1e15,catalogue_true.catalogue["z"][i],cosmology,c=1.177,Delta=500.,type=type)
        y0_scal_rel[i] = cluster.get_y_norm(type="centre")

    q_scal_rel = catalogue_true.catalogue["q_opt"]*y0_scal_rel/catalogue_true.catalogue["y0"]
    ret = q_scal_rel

    if get_y == True:

        ret = (q_scal_rel,y0_scal_rel)

    return ret



class catalogue_planck:

    def __init__(self,type="mmf3",threshold=6.):

        fit_union = fits.open('/Users/user/Desktop/catalogues/HFI_PCCS_SZ-union_R2.08.fits')
        fit_mmf3 = fits.open('/Users/user/Desktop/catalogues/HFI_PCCS_SZ-MMF3_R2.08.fits')

        data_union = fit_union[1].data
        data_mmf3 = fit_mmf3[1].data

        if type == "mmf3":

            indices_mmf3 = np.where(data_mmf3["SNR"] > threshold)
            indices_union = data_mmf3["INDEX"][indices_mmf3]-1

        elif type == "cosmo":

            indices_mmf3 = []
            indices_union = []

            for i in range(0,len(data_mmf3["SNR"])):

                if (data_mmf3["SNR"][i] > threshold) and (data_union["COSMO"][data_mmf3["INDEX"][i]-1] == True):

                    indices_union.append(data_mmf3["INDEX"][i]-1)
                    indices_mmf3.append(i)

        self.catalogue = cluster_catalogue()

        self.catalogue.catalogue["q_opt"] = data_mmf3["SNR"][indices_mmf3]
        self.catalogue.catalogue["lon"] = data_union["GLON"][indices_union]
        self.catalogue.catalogue["lat"] = data_union["GLAT"][indices_union]
        self.catalogue.catalogue["z"] = data_union["REDSHIFT"][indices_union]


class catalogue_gcc_planck:

    def __init__(self,threshold=6.):

        cat_fits = fits.open('/Users/user/Desktop/catalogues/HFI_PCCS_GCC_R2.02.fits')
        data = cat_fits[1].data

        self.catalogue = cluster_catalogue()

        self.catalogue.catalogue["lon"] = data["GLON"]
        self.catalogue.catalogue["lat"] = data["GLAT"]
        self.catalogue.catalogue["flux_quality"] = data["FLUX_QUALITY"]


class catalogue_cs_planck:

    def __init__(self,threshold=6.):

        names = ['/Users/user/Desktop/catalogues/COM_PCCS_857_R2.01.fits',
        '/Users/user/Desktop/catalogues/COM_PCCS_545_R2.01.fits']

        lon = np.empty(0)
        lat = np.empty(0)

        for i in range(0,len(names)):

            cat_fits = fits.open(names[i])
            data = cat_fits[1].data

            lon = np.append(lon,data["GLON"])
            lat = np.append(lat,data["GLAT"])
        #    self.flux_quality = data["FLUX_QUALITY"]

        self.catalogue = cluster_catalogue()

        self.catalogue.catalogue["lon"] = lon
        self.catalogue.catalogue["lat"] = lat

#tnoi: "it" or "tnoi"
#modes: "fixed" or "find"
#name_cmmf: "mmf","cmmf","cmmf_beta","cmmf_betaT","cmmf_betaTbeta"
#nfreqs = 4, 5, 6

class catalogue_clusterdb:

    def __init__(self):

        name = '/Users/user/Desktop/catalogues/masterclusters.fits'

        cat_fits = fits.open(name)
        data = cat_fits[1].data

        print(data.dtype.names)

        self.ra = data["Ra"]
        self.dec = data["Dec"]

        gc = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree, frame='icrs')
        coords = gc.transform_to('galactic')

        self.lon = np.zeros(len(self.ra))
        self.lat = np.zeros(len(self.ra))

        for i in range(0,len(self.ra)):

            self.lon[i] = coords[i].l.value
            self.lat[i] = coords[i].b.value

        self.redshift = data["Redshift"]

        self.catalogue = cluster_catalogue()

        self.catalogue.catalogue["lon"] = self.lon
        self.catalogue.catalogue["lat"] = self.lat
        self.catalogue.catalogue["z"] = self.redshift

class catalogue_act_dr5:

    def __init__(self):

        name = '/Users/user/Desktop/catalogues/DR5_cluster-catalog_v1.1.fits'
        cat_fits = fits.open(name)
        data = cat_fits[1].data

        self.ra = data["RADeg"]
        self.dec = data["decDeg"]
        self.redshift = data["redshift"]

        gc = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree, frame='icrs')
        coords = gc.transform_to('galactic')

        self.lon = np.zeros(len(self.ra))
        self.lat = np.zeros(len(self.ra))

        for i in range(0,len(self.ra)):

            self.lon[i] = coords[i].l.value
            self.lat[i] = coords[i].b.value

        self.catalogue = cluster_catalogue()

        self.catalogue.catalogue["lon"] = self.lon
        self.catalogue.catalogue["lat"] = self.lat
        self.catalogue.catalogue["z"] = self.redshift

def get_detection_cib_name(name,cib_random,name_cmmf,n_freqs,suffix=""):

    prename = "paper_extr_cib_nops_"

#    name = prename + tnoi + "_" + modes + "_apodold_"
    name = prename + name + "_"

    if cib_random == True:

        name = name + "CIB_random_"

    name = name + name_cmmf + "_" + str(n_freqs) + suffix + "_websky"

    return name

def get_all_detection_cib_names(suffix=""):

    names = [
    "it_fixed_apodold",
    #"it_find_apodold",
    ]
    cib_randoms = [True,False]
    names_cmmf_0 = ["mmf","cmmf","cmmf_beta","cmmf_betaT","cmmf_betaTbeta"]
    names_cmmf = [["mmf","cmmf"],names_cmmf_0,names_cmmf_0]
    n_freqs = [4,5,6]

    detection_names = []

    for k in range(2,len(n_freqs)):
    #for k in range(0,2):

        nfreq = n_freqs[k]

        for i in range(0,len(names)):

            for j in range(0,len(cib_randoms)):

                name = names[i]
                cib_random = cib_randoms[j]

                if cib_random == True:

                    name_cmmf = "mmf"
                    detection_names.append(get_detection_cib_name(name,cib_random,name_cmmf,nfreq,suffix=suffix))

                else:

                    for l in range(0,len(names_cmmf[k])):

                        name_cmmf = names_cmmf[k][l]
                        detection_names.append(get_detection_cib_name(name,cib_random,name_cmmf,nfreq,suffix=suffix))

    return detection_names

def get_all_detection_cib_robustness_names(channel):

    base = "paper_extr_cib_nops_it_fixed_apodold_cmmf"

    params = ["beta","T0"]
    params2 = ["beta","beta"]
    params_sigma = [-2,-1,1,2]

    detection_names = []


    for i in range(0,2):

        for j in range(0,len(params_sigma)):

            if channel == 6:

                name = base + "_" + params2[i] + "_" + str(channel) + "_robust_" + params[i] + "_" + str(params_sigma[j]) + "_websky"
                detection_names.append(name)

            elif channel == 4:

                name = base + "_" + str(channel) + "_robust_" + params[i] + "_" + str(params_sigma[j]) + "_websky"
                detection_names.append(name)

    return detection_names

def convert_catalogue(cat_old):

    if hasattr(cat_old,'q_opt'):

        cat_new = cluster_catalogue()

        cat_new.catalogue["q_opt"] = cat_old.q_opt
        cat_new.catalogue["y0"] = cat_old.y0
        cat_new.catalogue["theta_500"] = cat_old.theta_500
        cat_new.catalogue["theta_x"] = cat_old.theta_x
        cat_new.catalogue["theta_y"] = cat_old.theta_y
        cat_new.catalogue["lon"] = cat_old.lon
        cat_new.catalogue["lat"] = cat_old.lat
        cat_new.catalogue["m_500"] = cat_old.m_500
        cat_new.catalogue["z"] = cat_old.z
        cat_new.catalogue["pixel_ids"] = cat_old.pixel_ids

    else:

        cat_new = cat_old

    return cat_new

def convert_results(results_old):

    results_old.results_grid = convert_catalogue(results_old.results_grid)
    results_old.results_refined = convert_catalogue(results_old.results_refined)
    results_old.results_grid_noit = convert_catalogue(results_old.results_grid_noit)
    results_old.results_refined_noit = convert_catalogue(results_old.results_refined_noit)
    results_old.results_true = convert_catalogue(results_old.results_true)

    #return results_old

def save_object(filename,obj):

    with open(filename, 'wb') as outp:  # Overwrites any existing file.

        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
