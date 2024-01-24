import numpy as np
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
from szifi import maps

def get_cutout(hp_map,lonlat,nx,l,coord_type="G"):

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

    for i in range(0,len(mask_tile)):

        neighbours = hp.pixelfunc.get_all_neighbours(nside_map,i)

        if len(set(mask_tile[neighbours])) == 1.:

            mask_tile[i] = 0.

        else:

            mask_tile[i] = 1.

    return mask_tile

def get_so_mask(nside):

    mask = np.zeros(hp.pixelfunc.nside2npix(nside))
    pixel_ids = np.arange(hp.pixelfunc.nside2npix(nside))

    dec_min = -63. #in degrees
    dec_max = 23. #in degrees

    glon,glat = hp.pixelfunc.pix2ang(nside,pixel_ids,lonlat=True)
    c = SkyCoord(l=glon*u.degree, b=glat*u.degree, frame='galactic')
    icrs = c.icrs
    #    ra = icrs.ra.value
    dec = icrs.dec.value

    indices = np.where((dec > dec_min) & (dec < dec_max))
    mask[indices] = 1.

    return mask

def glonglat_to_radec(glon,glat):

    c = SkyCoord(l=glon*u.degree, b=glat*u.degree, frame='galactic')
    icrs = c.icrs

    ra = icrs.ra.value
    dec = icrs.dec.value

    return ra,dec

def bin_radially(map,lon,lat,theta_bins_edges_rad):

    nside_factor = 2**7

    nside = hp.npix2nside(len(map))
    nside_coarse = nside//nside_factor
    npix_coarse = hp.nside2npix(nside_coarse)
    map_coarse = np.zeros(npix_coarse)
    pix_ids = np.arange(npix_coarse)
    pix_coords = hp.pix2ang(nside_coarse,pix_ids,lonlat=True)
    pix_distances = hp.rotator.angdist((lon,lat),pix_coords,lonlat=True)

    buffer = np.sqrt(hp.nside2pixarea(nside_coarse))
    pix_indices = np.where(pix_distances < theta_bins_edges_rad[-1]+buffer)
    map_coarse[pix_indices] = 1.
    map_indices = hp.pixelfunc.ud_grade(map_coarse,nside)
    indices = np.where(map_indices == 1.)[0]

    (lon_pixels,lat_pixels) = hp.pix2ang(nside,indices,lonlat=True)
    distances = hp.rotator.angdist((lon,lat),(lon_pixels,lat_pixels),lonlat=True)

    theta_bins = (theta_bins_edges_rad[0:-1] + theta_bins_edges_rad[1:])*0.5
    binned_map = np.zeros(len(theta_bins))

    for i in range(0,len(binned_map)):

        indices_bin = np.where((distances < theta_bins_edges_rad[i+1]) & (distances > theta_bins_edges_rad[i]))[0]
        map_selected = map[indices][indices_bin]
        binned_map[i] = np.mean(map_selected)

    return theta_bins,binned_map

def get_tile_fsky(nside_tile,mask):

    n_tile = hp.nside2npix(nside_tile)
    n_side_mask = hp.npix2nside(len(mask))

    skyfracs = np.zeros(n_tile)

    for i in range(0,n_tile):

        tile_map = np.zeros(n_tile)
        tile_map[i] = 1.
        tile_map_hd = hp.pixelfunc.ud_grade(tile_map,n_side_mask)

        tile_frac = np.sum(tile_map_hd*mask)/np.sum(tile_map_hd)

        skyfracs[i] = tile_frac
        print(i,tile_frac)

    return skyfracs/n_tile
