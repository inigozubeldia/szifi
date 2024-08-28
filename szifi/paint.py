import numpy as np
import pylab as pl
import healpy as hp
from .model import *
import time

class catalogue_painter:

    def __init__(self,catalogue,exp,pix,nside,cosmology):

        self.catalogue = catalogue #dictionary
        self.exp = exp #experiment instance
        self.pix = pix #pixelisation of cut-outs
        self.nside = nside #nside of output map
        self.cosmology = cosmology

    def paint_clusters(self,name,beam_type="gaussian"):

        n_freqs = len(self.exp.nu_eff)

        npix = hp.pixelfunc.nside2npix(self.nside)
        t_map_hp = np.zeros((npix,n_freqs))

        lon_pix,lat_pix = hp.pix2ang(self.nside,np.arange(npix),lonlat=True)

        lon = self.catalogue["lon"]
        lat = self.catalogue["lat"]
        m = self.catalogue["M"] #in solar masses
        redshift = self.catalogue["z"]
        theta_so = self.catalogue["theta_so"]
        n_clusters = len(lon)

        coords = hp.ang2vec(lon,lat,lonlat=True)
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        x_pix,y_pix,z_pix = hp.pix2vec(self.nside,np.arange(npix))

        print("N clusters",n_clusters)

        for i in range(0,n_clusters):

            t0 = time.time()

            print(i)

            cos = x[i]*x_pix + y[i]*y_pix + z[i]*z_pix
            dist = np.arccos(cos)
            indices = np.where((dist < theta_so[i]*5.) & (0 <= dist) & (cos > 0.))[0]

            print("Mass",m[i])

            cluster = gnfw(m[i]/1e15,redshift[i],self.cosmology,type="arnaud")

            theta,tsz_vec = cluster.get_t_vec_convolved_hankel(self.pix,self.exp,beam_type=beam_type)

            for j in range(0,n_freqs):

                t_map_hp[indices,j] = t_map_hp[indices,j] + np.interp(dist[indices],theta,tsz_vec[:,j])

            t1 = time.time()

            print("Time",t1-t0)

        np.save(name,t_map_hp)
