## Functions for taking flat-sky cutouts of a CAR map
## We tesselate the sky with HEALPIX but project straight from CAR to locally conformal projection
import healpy as hp
import numpy as np
from pixell import enmap, wcsutils, reproject, curvedsky
import pixell
from pkg_resources import parse_version
if parse_version(pixell.__version__) < parse_version('0.23.8'):
    raise ImportError("pixell >= 0.23.8 required for CAR projections")

### Utils ###
def make_vprint(verbose_level):
    def vprint(x, vlevel=1, return_only=False):
        if verbose_level >= vlevel:
            if not return_only:
                print(x)
            return x
        else:
            return ""
    return vprint

def healpix2radec(nside, ipix):
    """Convert healpix pixel id to RA (-pi, pi], Dec [-pi/2, pi/2]"""
    theta, phi = hp.pix2ang(nside, ipix) # Colat, lon, rads
    theta = np.pi/2 - theta # Convert colat to lat/dec
    phi -= (2*np.pi) * (phi > np.pi) # RA>pi wrapped to -pi
    return phi, theta

def get_target_wcs(dx_deg, field_shape, center_radec_deg):
    """Define a wcs object
       dx_deg: pixel size in degrees
       field_shape: 2-tuple shape of field in pixels, (dec, RA) or (y,x)
       center_radec_deg: 2-tuple center of the field in RA, dec, degrees
    """
    wcs = wcsutils.explicit(ctype=["RA---CAR", "DEC--CAR"], cdelt=[-dx_deg, dx_deg],
                            crpix=np.array(field_shape)[::-1]/2, crval=center_radec_deg)
    return wcs

def get_field_shape(field_size_deg, dx_deg, nside=None, min_buffer_deg=None):
    """Get the shape of a SQUARE field in (pixels, pixels)
    field_size_deg: side of the square in degrees, will be rounded up to nearest pixel
    dx_deg: pixel size in degrees
    nside:  If field_size_deg=None, used to determine min size to hold a healpixel of this nside
    min_buffer_deg: If field_size_deg=None, number of degrees buffer to add to field outside edge of healpixel
    """
    assert (field_size_deg is None) or np.isscalar(field_size_deg)
    assert np.isscalar(dx_deg)
    if field_size_deg is None:
        mindist = hp.max_pixrad(nside) # max dist between pixel center and corner
        field_size_deg = mindist + 2*min_buffer_deg
    npix = int(np.ceil(field_size_deg / dx_deg))
    field_shape = (npix, npix)
    return field_shape

### Project a tile ###
def project_car_spline(imap, target_geometry, dtype=None, order=3):
    """Project a map onto a CAR geometry with spline iterpolation"""
    nn, wcs = target_geometry
    proj = enmap.project(imap, nn, wcs, bsize=max(nn)+1, order=order)
    return proj

def project_car_sht(alms, target_geometry, dtype, order=None):
    """Project a map onto a CAR geometry using SHTs"""
    nn, wcs = target_geometry
    proj = enmap.zeros(*target_geometry, dtype)
    curvedsky.alm2map(alms, proj)
    return proj

def make_healpix_tiles_car(nside, car_geometry, dtype=None, order=None):
    """Make a car map of healpixels valued with their healpixel index in RING"""
    full_shape, wcs = car_geometry
    tiles_hp = np.arange(hp.nside2npix(nside))
    tiles_car = reproject.healpix2map(tiles_hp, full_shape, wcs, method="spline", order=0)
    return tiles_car

def make_tile(imap, dx_deg, field_shape, center_radec_deg, project_func, dtype=None, order=None, maskval=None, savename=None):
    """Project onto a tile
    imap: map to project. Can be healpix, enmap, or alms depending on project_func
    dx_deg: pixel size in degrees
    field_shape: (Ny, Nx) field shape in pixels
    center_radec_deg: (RA, dec) in degrees of the field center
    project_func: function to use for projection; project_car_sht, project_car_spline, make_healpix_tiles_car
    dtype: Only used/needed for project_car_sht
    order: Order of the spline interpolation, only used for that
    maskval: Apply a (tile==maskval) cut before returning; used for making a tile mask
    savename: filename to save to
    Return: enmap tile
    """
    target_wcs = get_target_wcs(dx_deg, field_shape, center_radec_deg)
    tile = project_func(imap, (field_shape, target_wcs), dtype, order)
    if maskval is not None:
        tile = (tile == maskval)
    if savename is not None:
        np.save(savename, tile)
    return tile

### Project many tiles ###
def make_all_masks(nside, dx_deg, field_shape, healpix_ids, savename_prefix='map', verbosity=1, imin=0, imax=int(1e10)):
    """Make tile masks for a map. See make_all_tiles for params"""
    vprint = make_vprint(verbosity)
    fsize = len(str(np.max(healpix_ids))) # Max number of digits in id, for the filename
    ra_deg, dec_deg = np.rad2deg(healpix2radec(nside, healpix_ids))
    ## Make tile masks
    vprint("Making tile masks")
    for ii, field_id in enumerate(healpix_ids[np.arange(imin,min(imax, healpix_ids.size))]):
        tilemaskname = savename_prefix + f"_tile{field_id:0{fsize}d}_tilemask.npy"
        make_tile(nside, dx_deg, field_shape, (ra_deg[ii], dec_deg[ii]), make_healpix_tiles_car, maskval=field_id, savename=tilemaskname)
        vprint(f"{ii+1}/{len(healpix_ids)}", 2)

def make_all_tiles(imap, nside, dx_deg, field_shape, healpix_ids, projection_method, sht_lmax=None, alm_fn=None, order=3, savename_prefix='map', verbosity=1, imin=0, imax=int(1e10)):
    """Project a CAR map onto tiles defined by healpixels
    imap: map to project
    nside: NSIDE of healpix map defining the logical tiles
    dx_deg: pixel size in degrees
    field_shape: (Ny, Nx) field shape in pixels
    healpix_ids: list[int] of the healpixel numbers (in RING) defining the tiles
    projection_method: 'spline' or 'sht'
    sht_lmax: Max ell for the SHT
    alm_fn: Optional .npy file from which to read alms for SHT
    order: Spline interpolation order
    savename_prefix: prefix for savenames
    verbosity: 0,1,2, how much to print
    imin, imax: ints, limit how many tiles to do
    """
    vprint = make_vprint(verbosity)
    fsize = len(str(np.max(healpix_ids))) # Max number of digits in id, for the filename
    ra_deg, dec_deg = np.rad2deg(healpix2radec(nside, healpix_ids))
    ## Make tiles
    if healpix_ids is None:
        tiles_car = make_healpix_tiles_car(nside, imap.geometry)
        healpix_ids = np.unique(tiles_car)
        del tiles_car

    if projection_method == 'spline':
        dtype=None
        project_fn = project_car_spline
    elif projection_method == 'sht':
        dtype = imap.dtype
        project_fn = project_car_sht
        if alm_fn is None:
            vprint("Taking SHT of map")
            imap = curvedsky.map2alm(imap, lmax=sht_lmax)
        else:
            imap = np.load(alm_fn)
    else:
        raise ValueError(f"projection_method={projection_method}; Only 'spline' and 'sht' allowed")
    vprint("Making tiles")
    for ii, field_id in enumerate(healpix_ids[np.arange(imin,min(imax, healpix_ids.size))]):
        tilename = savename_prefix + f"_tile{field_id:0{fsize}d}.npy"
        make_tile(imap, dx_deg, field_shape, (ra_deg[ii], dec_deg[ii]), project_fn, dtype, order, None, tilename)
        vprint(f"{ii+1}/{len(healpix_ids)}", 2)
