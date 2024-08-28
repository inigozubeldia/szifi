import numpy as np
import pylab as pl
import healpy as hp
import scipy.integrate as integrate
import scipy.special as sp
from random import randrange
import copy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from astropy.io import fits
from .model import *
from .maps import *
from .sphere import *

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

        lon,lat = get_lonlat(i,self.catalogue["theta_x"]-pix.nx*pix.dx*0.5,self.catalogue["theta_y"]-pix.nx*pix.dx*0.5,n_side)
        self.catalogue["lon"] = lon
        self.catalogue["lat"] = lat

    def select_tile(self,i,pix,type="field",nside=8): #it works

        theta_x,theta_y = get_xy(i,self.catalogue["lon"],self.catalogue["lat"],nside)

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

def merge_detections(catalogue,radius_arcmin=10.,return_merge_flag=False,mode="closest",
fac_theta_500=1.,merge_radius_type="theta_500"):

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
            dist = get_angdist(catalogue_compare.catalogue["lon"][0]*np.ones(n_clusters),catalogue_compare.catalogue["lon"],
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

    elif mode == "masking":

        indices_sorted = np.argsort(catalogue.catalogue["q_opt"])[::-1]
        catalogue = get_catalogue_indices(catalogue,indices_sorted)

        i = 1

        while i > 0:

            if i > 1:

                catalogue_new = get_catalogue_indices(catalogue,[0])
                catalogue_merged.append(catalogue_new,append_keys="new")

                lon1 = catalogue_new.catalogue["lon"][0]*np.ones(len(catalogue.catalogue["lon"]))
                lat1 = catalogue_new.catalogue["lat"][0]*np.ones(len(catalogue.catalogue["lat"]))
                lon2 = catalogue.catalogue["lon"]
                lat2 = catalogue.catalogue["lat"]
                coords1 = [lon1,lat1]
                coords2 = [lon2,lat2]

                distances = get_distance_sphere_lonlat(coords1,coords2)

                if merge_radius_type == "fixed":

                    indices_remove = np.where(distances < fac_theta_500)[0]

                elif merge_radius_type == "theta_500":

                    indices_remove = np.where(distances < fac_theta_500*catalogue_new.catalogue["theta_500"][0])[0]

                catalogue = remove_catalogue_indices(catalogue,indices_remove)

            elif i == 1:

                catalogue_new = get_catalogue_indices(catalogue,[0])
                catalogue_merged.append(catalogue_new,append_keys="new")
                indices_remove = [0]
                catalogue = remove_catalogue_indices(catalogue,indices_remove)

            i = len(catalogue.catalogue["lon"])

            if i == 0:

                break

        #    print(i,len(indices_remove),catalogue_merged.catalogue["q_opt"])

        ret = catalogue_merged

    elif mode == "masking_max":

        indices_sorted = np.argsort(catalogue.catalogue["q_opt"])[::-1]
        catalogue = get_catalogue_indices(catalogue,indices_sorted)

        i = 1

        while i > 0:

            if i > 1:

                catalogue_new = get_catalogue_indices(catalogue,[0])
                catalogue_merged.append(catalogue_new,append_keys="new")

                lon1 = catalogue_new.catalogue["lon"][0]*np.ones(len(catalogue.catalogue["lon"]))
                lat1 = catalogue_new.catalogue["lat"][0]*np.ones(len(catalogue.catalogue["lat"]))
                lon2 = catalogue.catalogue["lon"]
                lat2 = catalogue.catalogue["lat"]
                coords1 = [lon1,lat1]
                coords2 = [lon2,lat2]

                distances = get_distance_sphere_lonlat(coords1,coords2)

                masking_radius = np.max([fac_theta_500*catalogue_new.catalogue["theta_500"][0],radius_arcmin])
                indices_remove = np.where(distances < masking_radius)[0]

                catalogue = remove_catalogue_indices(catalogue,indices_remove)

            elif i == 1:

                catalogue_new = get_catalogue_indices(catalogue,[0])
                catalogue_merged.append(catalogue_new,append_keys="new")
                indices_remove = [0]
                catalogue = remove_catalogue_indices(catalogue,indices_remove)

            i = len(catalogue.catalogue["lon"])

            if i == 0:

                break

        #    print(i,len(indices_remove),catalogue_merged.catalogue["q_opt"])

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

    n_clusters_1 = catalogue_1.get_n_clusters()

    for i in range(0,n_clusters_1):

        print(i,"of",n_clusters_1)

        if lonlat == False:

            dist = np.sqrt((catalogue_1.catalogue["theta_x"][i]-catalogue_2.catalogue["theta_x"])**2+(catalogue_1.catalogue["theta_y"][i]-catalogue_2.catalogue["theta_y"])**2)

        elif lonlat == True:

            dist = get_angdist(catalogue_1.catalogue["lon"][i],catalogue_2.catalogue["lon"],catalogue_1.catalogue["lat"][i],catalogue_2.catalogue["lat"])

        if unique == False:

            indices_closest = np.where(dist < id_radius_arcmin/180./60.*np.pi)[0]

        elif unique == True:

            indices_closest = np.where((dist < id_radius_arcmin/180./60.*np.pi) & (indices_catalogue_2 == 0.))[0]

        if len(indices_closest) != 0:

            print(indices_closest)

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

    i,j = get_ij_from_theta(cat.catalogue["theta_x"],cat.catalogue["theta_y"],pix)
    mask_values = mask_select[i,j]
    indices = np.where(mask_values != 0.)[0]

    return get_catalogue_indices(cat,indices)

def apply_mask_select_fullsky(cat,mask_select):

    nside = hp.npix2nside(len(mask_select))
    pixs = hp.ang2pix(nside,cat.catalogue["lon"],cat.catalogue["lat"],lonlat=True)
    mask_values = mask_select[pixs]
    indices = np.where(mask_values != 0.)[0]

    return get_catalogue_indices(cat,indices)

def get_indices_mask_select_fullsky(cat,mask_select):

    nside = hp.npix2nside(len(mask_select))
    pixs = hp.ang2pix(nside,cat.catalogue["lon"],cat.catalogue["lat"],lonlat=True)
    mask_values = mask_select[pixs]
    indices = np.where(mask_values != 0.)[0]

    return indices

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

            cat_new.catalogue[key] = np.array(cat_old.catalogue[key])[indices]

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
        distances = get_angdist(cat_true.catalogue["lon"],cat_obs.catalogue["lon"],cat_true.catalogue["lat"],cat_obs.catalogue["lat"])/np.pi*60.*180.
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

def get_completeness_err(bins_edges,catalogue_true,catalogue_obs,q_th=4.5,n_boots=100,type="std"):

    completeness_boots = np.zeros((n_boots,len(bins_edges)-1))

    for i in range(0,n_boots):

        n_clusters = len(catalogue_true.catalogue["q_opt"])
        indices = np.random.randint(0,n_clusters,size=n_clusters)
        cat_true_boots = get_catalogue_indices(catalogue_true,indices)
        cat_obs_boots = get_catalogue_indices(catalogue_obs,indices)
        comparison = catalogue_comparer(cat_true_boots,cat_obs_boots)
        bins,completeness,completeness_err = comparison.get_completeness(bins_edges,q_th=q_th)
        completeness_boots[i,:] = completeness

    if type == "std":

        completeness_err = np.std(completeness_boots,axis=0)
        ret = bins,completeness_err

    elif type == "quantile":

        completeness_low = np.quantile(completeness_boots,0.5-0.341,axis=0)
        completeness_high = np.quantile(completeness_boots,0.5+0.341,axis=0)
        ret = bins,completeness_low,completeness_high

    return ret


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

class results_detection:

    def __init__(self):

        self.catalogues = {}
        self.sigma_matrix = None
        self.sigma_matrix_noit = None
        self.theta_500_vec = None
        self.sigma_vec = {}

        self.f_sky = None
        self.info = None


    def set_pixel_ids(self,pixel_ids):

        for key in self.catalogues.keys():

            self.catalogues[key].catalogue["pixel_ids"] = pixel_ids

    def get_lonlat(self,i,pix,nside=8):

        for key in self.catalogues.keys():

            self.catalogues[key].get_lonlat(nside,i,pix)

    def append(self,results_new):

        for key in results_new.catalogues.keys():

                if key in self.catalogues:


                    self.catalogues[key].append(results_new.catalogues[key],append_keys="new")

                else:

                    self.catalogues[key] = results_new.catalogues[key]


    def make_copy(self):

        results_copy = results_detection()
        results_copy.sigma_vec = self.sigma_vec
        results_copy.sigma_vec_noit = self.sigma_vec_noit
        results_copy.theta_500_sigma = self.theta_500_sigma
        results_copy.f_sky = self.f_sky
        results_copy.info = self.info

        for key in self.catalogues.keys():

            results_copy.catalogues[key] = self.catalogues[key]


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

        theta_x,theta_y = get_theta_from_ij(ix,jx,pix)
        theta_x_vec[i] = theta_x
        theta_y_vec[i] = theta_y

    return theta_x_vec,theta_y_vec

#catalogue_true mustn't have -1's

class master_catalogue:

    def __init__(self,catalogue,catalogue_name,id_radius_arcmin=None,id_mode=None,
    unique=None,sort=None,masks=None,find_label=True):

        #catalogue should have no -1 values

        self.id_radius_arcmin = id_radius_arcmin
        self.id_mode = id_mode
        self.unique = unique
        self.sort = sort
        self.masks = masks
        self.find_label = find_label

        self.sigma_matrices = {}
        self.skyfracs = {}

        self.catalogue_master = copy.deepcopy(catalogue)

        for key in catalogue.catalogue.keys():

            if len(catalogue.catalogue[key]) == 0:

                self.catalogue_master.catalogue[key] = -np.ones(len(catalogue.catalogue["q_opt"]))

        for key in catalogue.catalogue.keys():

            self.catalogue_master.catalogue[key + "_" + catalogue_name] = self.catalogue_master.catalogue[key]

        self.indices = np.arange(len(catalogue.catalogue["q_opt"]))
        self.catalogue_master.catalogue["index"] = self.indices
        self.individual_catalogues = [catalogue_name]

    def add_catalogue(self,catalogue_in,catalogue_name_in):

        self.individual_catalogues.append(catalogue_name_in)

        if self.find_label is True:

            catalogue_master_id,catalogue_in_id = identify_clusters(self.catalogue_master,catalogue_in,
            lonlat=True,id_radius_arcmin=self.id_radius_arcmin,mode=self.id_mode,unique=self.unique,sort=self.sort)

        elif self.find_label is False:

            catalogue_master_id = self.catalogue_master
            catalogue_in_id = catalogue_in

        indices_null = np.where(catalogue_master_id.catalogue["index"] == -1)[0]
        indices_pos = np.where(catalogue_master_id.catalogue["index"] > -1)[0]
        indices_sort = indices_pos[np.argsort(catalogue_master_id.catalogue["index"][indices_pos].astype(int))]

        for key in catalogue_in_id.catalogue.keys():

            if len(catalogue_in_id.catalogue[key]) > 0:

                self.catalogue_master.catalogue[key + "_" + catalogue_name_in] = catalogue_in_id.catalogue[key][indices_sort]

        indices_master_new = np.arange(self.catalogue_master.catalogue["index"][-1]+1,self.catalogue_master.catalogue["index"][-1]+len(indices_null)+1,dtype=np.int64)

        for key in self.catalogue_master.catalogue.keys():

            self.catalogue_master.catalogue[key] = np.append(self.catalogue_master.catalogue[key],-np.ones(len(indices_master_new)))

        self.catalogue_master.catalogue["index"][indices_master_new] = indices_master_new

        for key in catalogue_in_id.catalogue.keys():

            if len(catalogue_in_id.catalogue[key]) > 0:

                self.catalogue_master.catalogue[key + "_" + catalogue_name_in][indices_master_new] = catalogue_in_id.catalogue[key][indices_null]
                self.catalogue_master.catalogue[key][indices_master_new] = catalogue_in_id.catalogue[key][indices_null]

    def apply_mask(self,mask):

        self.catalogue_master = apply_mask_select_fullsky(self.catalogue_master,mask)

    def add_mask_index(self,mask):

        self.catalogue_master.catalogue["cosmology_mask"] = -np.ones(len(self.catalogue_master.catalogue["lon"]))

        indices = get_indices_mask_select_fullsky(self.catalogue_master,mask)
        self.catalogue_master.catalogue["cosmology_mask"][indices] = np.ones(len(indices))

class detection_processor:

    def __init__(self,results_dict,params_szifi):

        self.results = results_detection()

        field_ids = results_dict.keys()
        n_theta = len(params_szifi["theta_500_vec_arcmin"])

        self.sigma_matrix = np.zeros((len(field_ids),n_theta))
        self.sigma_matrix_noit = np.zeros((len(field_ids),n_theta))
        self.theta_vec_matrix = np.zeros((len(field_ids),n_theta))

        i = 0

        for field_id in field_ids:

            if "find_1" in results_dict[field_id].sigma_vec:

                self.sigma_matrix[i,:] = results_dict[field_id].sigma_vec["find_1"]

            if "find_0" in results_dict[field_id].sigma_vec:

                self.sigma_matrix_noit[i,:] = results_dict[field_id].sigma_vec["find_0"]

            if params_szifi["iterative"] == True and "find_1" not in results_dict[field_id].sigma_vec:

                cat_keys = copy.deepcopy(list(results_dict[field_id].catalogues.keys()))

                for cat_key in cat_keys:

                #    print(field_id,cat_key)

                    results_dict[field_id].catalogues[cat_key[0:-1] + "1"] = results_dict[field_id].catalogues[cat_key]
                    del results_dict[field_id].catalogues[cat_key]
            self.results.append(results_dict[field_id])

        #    print(results_dict[field_id].catalogues)

            i = i + 1

        self.results.sigma_matrix_noit = self.sigma_matrix_noit
        self.results.sigma_matrix = self.sigma_matrix
        self.results.theta_500_vec = params_szifi["theta_500_vec_arcmin"]

    def apply_mask(self,mask): #full sky implementation

        for key in self.results.catalogues.keys():

            self.results.catalogues[key] = apply_mask_select_fullsky(self.results.catalogues[key],mask)
