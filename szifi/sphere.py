import numpy as np
import pylab as pl
import healpy as hp
import mmf
import maps
import expt

class planck_maps:

    def __init__(self,freqs=np.arange(6)):

        self.tmaps = []
        self.freqs = freqs
        self.nside = 2048

        for i in freqs:

            self.tmaps.append(load_planck_tmap(i))

    def get_tmap(self,i):

        print(np.where(self.freqs==i)[0])

        return self.tmaps[np.where(self.freqs==i)[0]]

class websky_map:

    def __init__(self,component,freqs=np.arange(6)):

        self.tmaps = []
        self.freqs = freqs
        self.nside = 2048
        exp = expt.websky_specs()

        for i in freqs:

            self.tmaps.append(hp.read_map("/Users/user/Desktop/data/websky/" + component + "_" + str(exp.nu_eff_GHz[i])+ "_2048.fits"))

    def get_tmap(self,i):

        print(np.where(self.freqs==i)[0])

        return self.tmaps[np.where(self.freqs==i)[0]]


def load_planck_tmap(i):

    freqs = ["100","143","217","353-psb","545","857"]

    return hp.read_map("/Users/user/Desktop/data/planck_data/HFI_SkyMap_" + str(freqs[i]) + "_2048_R3.01_full.fits",field=0)

#Galaxy mask: 20, 40, 60, 70, 80, 90, 97 and 99 of sky unmasked

class planck_mask:

    def __init__(self,field=2):

        self.galaxy_mask = hp.read_map('/Users/user/Desktop/data/planck_data/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=field)
        self.point_mask = np.ones(len(self.galaxy_mask))

        for i in range(0,6):

            self.point_mask *= hp.read_map('/Users/user/Desktop/data/planck_data/HFI_Mask_PointSrc_2048_R2.00.fits',field=i)

class flat_fields:

#l in deg

    def __init__(self,maps,masks,nx=1024,l=14.8):

        self.maps = maps
        self.masks = masks
        self.nside_tile = 8
        self.n_tile = hp.nside2npix(self.nside_tile)
        print("N tiles",self.n_tile)
        self.mask_galaxy = self.masks.galaxy_mask
        self.mask_point = self.masks.point_mask
        self.npix = hp.nside2npix(self.maps.nside)
        self.nx = nx
        self.l = l
        print("Pixel size",l/nx*60.)


    def get_field(self,i,save=True,save_name=None):

        lon,lat = hp.pix2ang(self.nside_tile,i,lonlat=True)

        mask_tile = np.zeros(self.n_tile)
        mask_tile[i] = 1.
        nside_map = self.maps.nside
        mask_tile = hp.pixelfunc.ud_grade(mask_tile,nside_map)

        mask_tile_cutout = get_cutout(mask_tile,[lon,lat],self.nx,self.l)
        mask_galaxy = get_cutout(self.mask_galaxy,[lon,lat],self.nx,self.l)
        mask_point = get_cutout(self.mask_point,[lon,lat],self.nx,self.l)
        tmap = np.zeros((self.nx,self.nx,len(self.maps.tmaps)))

        for j in range(0,len(self.maps.tmaps)):

            tmap[:,:,j] = get_cutout(self.maps.tmaps[j],[lon,lat],self.nx,self.l)

        if save == True:

            np.save(save_name + str(i)+ "_tmap.npy",[tmap])
            np.save(save_name + str(i)+ "_mask.npy",[mask_galaxy,mask_point,mask_tile_cutout])


            for j in range(0,len(self.maps.tmaps)):

                f = pl.figure()
                pl.imshow(tmap[:,:,j],interpolation='none')
                f.savefig(save_name + str(i) + "_" + str(j) + ".pdf")
                f.clear()
                pl.close(f)
                f = pl.figure()
                pl.imshow(tmap[:,:,j]*mask_galaxy*mask_point,interpolation='none')
                f.savefig(save_name + str(i) + "_" + str(j) + "_masked.pdf")
                f.clear()
                pl.close(f)
                f = pl.figure()
                pl.imshow(tmap[:,:,j]*mask_galaxy*mask_point*mask_tile_cutout,interpolation='none')
                f.savefig(save_name + str(i) + "_" + str(j) + "_tmasked.pdf")
                f.clear()
                pl.close(f)

        return tmap,mask_galaxy,mask_point,mask_tile_cutout

    def get_all_fields(self,save=True,save_name=None,i_min=0):

        for i in range(i_min,self.n_tile):

            print(i)
            tmap,mask_galaxy,mask_point,mask_tile_cutout = self.get_field(i,save=save,save_name=save_name)

        """
        pl.imshow(tmap[:,:,0])
        pl.show()
        pl.imshow(tmap[:,:,0]*mask_map)
        pl.show()
        pl.imshow(tmap[:,:,0]*mask_map*mask_tile)
        pl.show()
        """

class flat_fields_websky:

#l in deg
#possible components: tsz, ksz, cib, cmb, synchro, dust

    def __init__(self,component,nx=1024,l=14.8):

        self.maps = websky_map(component)
        self.nside_tile = 8
        self.n_tile = hp.nside2npix(self.nside_tile)
        self.npix = hp.nside2npix(self.maps.nside)
        self.nx = nx
        self.l = l
        self.save_name = "/Users/user/Desktop/maps/websky_maps/websky"
        self.component = component

    def get_field(self,i,save=True,save_name="/Users/user/Desktop/maps/websky_maps/t_maps/"):

        lon,lat = hp.pix2ang(self.nside_tile,i,lonlat=True)

        tmap = np.zeros((self.nx,self.nx,len(self.maps.tmaps)))

        for j in range(0,len(self.maps.tmaps)):

            tmap[:,:,j] = get_cutout(self.maps.tmaps[j],[lon,lat],self.nx,self.l)

        if save == True:

            np.save(save_name + "_" + self.component + "_" + str(i) + "_tmap.npy",[tmap])

            for j in range(0,len(self.maps.tmaps)):

                f = pl.figure()
                pl.imshow(tmap[:,:,j],interpolation='none')
                f.savefig(save_name + "_" + self.component + "_" + str(i) + "_" + str(j) + ".pdf")
                f.clear()
                pl.close(f)

        return tmap

    def get_all_fields(self,save=True,save_name="/Users/user/Desktop/maps/websky_maps/t_maps/",i_min=0,i_max=None):

        if i_max is None:

            i_max = self.n_tile

        for i in range(i_min,i_max):

            print(i)
            tmap = self.get_field(i,save=save,save_name=save_name)

        """
        pl.imshow(tmap[:,:,0])
        pl.show()
        pl.imshow(tmap[:,:,0]*mask_map)
        pl.show()
        pl.imshow(tmap[:,:,0]*mask_map*mask_tile)
        pl.show()
        """


def get_cutout(hp_map,lonlat,nx,l):

    [lon,lat] = lonlat
    l2 = l/2.
    lonmin = -l2
    lonmax = l2
    latmin = -l2
    latmax = l2

    cutout_map = hp.visufunc.cartview(hp_map,rot=(lon,lat,0.),
    xsize=nx,lonra=[lonmin,lonmax],latra=[latmin,latmax],return_projected_map=True).data
    cutout_map = np.flip(cutout_map,axis=0)

    return cutout_map


class cutouts:

    def __init__(self,name):

        self.name = name
        self.nx = 1024
        self.l = 14.8  #in deg
        self.dx = self.l/self.nx/180.*np.pi
        self.nside_tile = 8
        self.n_tile = hp.nside2npix(self.nside_tile)

    def get_cutout_i(self,i):

        [tmap] = np.load(self.name + str(i)+ "_tmap.npy")
        [mask_galaxy,mask_point,mask_tile] = np.load(self.name + str(i)+ "_mask.npy")
        cutout_i = cutout(tmap,mask_galaxy,mask_point,mask_tile)

        return cutout_i

#tmap is in muK

class cutout:

    def __init__(self,tmap,mask_galaxy,mask_point,mask_tile):

        factor = 1./np.array([1.,1.,1.,1.,58.04,2.27])# np.array([1e6,1e6,1e6,1e6,1e6,1e6])/np.array([1.,1.,1.,1.,58.04,2.27])
        self.tmap = tmap

        for i in range(0,6):

            self.tmap[:,:,i] = self.tmap[:,:,i]*factor[i]

        self.mask_galaxy = mask_galaxy
        self.mask_point = mask_point
        self.mask_tile = mask_tile

def get_field_limits(n_side,pixel_id,nx,dx):

    lon_c,lat_c = hp.pix2ang(n_side,pixel_id,lonlat=True)
    proj = hp.projector.CartesianProj(rot=(lon_c,lat_c,0.))

    theta_min = -nx*dx*0.5/(np.pi/180.)
    theta_max = nx*dx*0.5/(np.pi/180.)

    lon_bl,lat_bl = proj.xy2ang(theta_min,theta_min,lonlat=True)
    lon_br,lat_br = proj.xy2ang(theta_max,theta_min,lonlat=True)
    lon_tl,lat_tl = proj.xy2ang(theta_min,theta_max,lonlat=True)
    lon_tr,lat_tr = proj.xy2ang(theta_max,theta_max,lonlat=True)

    return (lon_bl,lat_bl,lon_br,lat_br,lon_tl,lat_tl,lon_tr,lat_tr)

#theta_x and theta_y in rad

def get_lonlat(i,theta_x,theta_y,n_side):

    theta_x = theta_x/(np.pi/180.)
    theta_y = theta_y/(np.pi/180.)
    lon_c,lat_c = hp.pix2ang(n_side,i,lonlat=True)
    proj = hp.projector.CartesianProj(rot=(lon_c,lat_c,0.))
    lon,lat = proj.xy2ang(theta_x,theta_y,lonlat=True)

    if isinstance(lon,float) == True:

        if lon < 0.:

            lon = lon + 360.

#        lon = np.array([lon])
    #    lat = np.array([lat])

    else:

        for i in range(0,len(lon)):

            if lon[i] < 0.:

                lon[i] = lon[i] + 360.

    return lon,lat

#theta_x, theta_y returned in rad
def get_xy(i,lon,lat,n_side):

    lon_c,lat_c = hp.pix2ang(n_side,i,lonlat=True)
    proj2 = hp.projector.CartesianProj(rot=(lon_c,lat_c,0.))
    theta_x,theta_y = proj2.ang2xy(lon,lat,lonlat=True)
    theta_x = theta_x*np.pi/180.
    theta_y = theta_y*np.pi/180.

    return theta_x,theta_y

def get_angdist(lon1,lon2,lat1,lat2): #distance given in rad, input in deg

    if isinstance(lon1,float) == True and isinstance(lat1,float) == True:

        angdist = np.zeros(len(lon2))

        for i in range(0,len(lon2)):

            angdist[i] = hp.rotator.angdist((lon1,lat1),(lon2[i],lat2[i]),lonlat=True)

    else:

        angdist = np.zeros(len(lon1))

        for i in range(0,len(lon1)):

            angdist[i] = hp.rotator.angdist((lon1[i],lat1[i]),(lon2[i],lat2[i]),lonlat=True)

    return angdist

def convolve_pixwin(map,nside):

    pix_hp = hp.sphtfunc.pixwin(nside)
    ell = np.arange(0,len(pix_hp))

    return hp.sphtfunc.smoothing(map,beam_window=pix_hp)

def get_white_noise_map_hp(nside,n_lev):

    sigma_pixel = n_lev/(180.*60./np.pi*np.sqrt(hp.pixelfunc.nside2pixarea(nside)))
    npix = hp.pixelfunc.nside2npix(nside)
    noise_map = np.random.standard_normal(npix)*sigma_pixel

    return noise_map

def get_white_noise_map_convolved_hp(nside,n_lev):

    return convolve_pixwin(get_white_noise_map_hp(nside,n_lev),nside)

def deprojector(map_projected,pix,i_tile,nside_tile,nside_map):

    n_tiles = hp.nside2npix(nside_tile)
    m = np.zeros(n_tiles)
    m[i_tile] = 1.
    m_up = hp.pixelfunc.ud_grade(m,nside_map)

    n_pixel = len(m_up)
    indices = np.where(m_up == 1.)[0]

    lon_c,lat_c = hp.pix2ang(nside_tile,i_tile,lonlat=True)
    proj = hp.projector.CartesianProj(rot=(lon_c,lat_c,0.))

    map_ret = np.zeros(n_pixel)
    counts = np.zeros(n_pixel)

    for i in range(0,pix.nx):

        for j in range(0,pix.ny):

            theta_x,theta_y = maps.get_theta_from_ij(i,j,pix)
            theta_x = (theta_x- pix.nx*pix.dx*0.5)/(np.pi/180.)
            theta_y = (theta_y- pix.ny*pix.dy*0.5)/(np.pi/180.)
            lon,lat = proj.xy2ang(theta_x,theta_y,lonlat=True)
            i_pix = hp.pixelfunc.ang2pix(nside_map,lon,lat,lonlat=True)
            map_ret[i_pix] = map_ret[i_pix] + map_projected[i,j]
            counts[i_pix] = counts[i_pix] + 1.

    indices = np.where(counts==0.)[0]
    map_ret[indices] = 0.
    map_ret = map_ret/counts


    return map_ret


def get_tile_boundaries_map(nside_tile,nside_map):

    mask_tile = np.arange(12*nside_tile**2)
    mask_tile = hp.pixelfunc.ud_grade(mask_tile,nside_map)

    #print("N pix",len(mask_tile))

    for i in range(0,len(mask_tile)):

        #print(i)

        neighbours = hp.pixelfunc.get_all_neighbours(nside_map,i)

        if len(set(mask_tile[neighbours])) == 1.:

        #if all(i == mask_tile[neighbours][0] for i in mask_tile[neighbours]):

            mask_tile[i] = 0.

        else:

            mask_tile[i] = 1.

    return mask_tile
