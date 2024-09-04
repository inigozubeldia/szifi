import os
from pixell import enmap
import numpy as np
import szifi.car as car
from multiprocessing import Pool

def get_healpix_ids(healpix_id_fn, map_filename=None, nside=None):
    if os.path.exists(str(healpix_id_fn)):
        return np.loadtxt(healpix_id_fn).astype('int64')
    
    map_geometry = enmap.read_fits_geometry(map_filename)
    tiles_car = car.make_healpix_tiles_car(nside, map_geometry)  ## Need to redo to project mask directly from healpix to desired field
    healpix_ids = np.unique(tiles_car)
    if healpix_id_fn is not None:
        np.savetxt(healpix_id_fn, healpix_ids)
    return healpix_ids

def get_cutout(map_filename, nside, dx_deg, field_shape, projection_method, apod_pix, do_masks, do_tiles, prefix_mask, prefix_tile, sht_lmax=None, alm_fn=None, order=3, healpix_id_fn=None, verbosity=1, imin=0, imax=int(1e10), flip=True):
    healpix_ids = get_healpix_ids(healpix_id_fn, map_filename, nside)
    if do_masks:
        car.make_all_masks(nside, dx_deg, field_shape, healpix_ids, prefix_mask, verbosity, imin, imax, flip)
    if do_tiles:
        imap = enmap.read_map(map_filename)
        if (apod_pix is not None) and apod_pix > 0:
            imap = enmap.apod(imap, apod_pix) # something more advanced needed if there is a non-trivial footprint mask
        car.make_all_tiles(imap, nside, dx_deg, field_shape, healpix_ids, projection_method, sht_lmax, alm_fn, order, prefix_tile, verbosity, imin, imax, flip)

def collect_freqs(fn_in_list, fn_out):
    maps = np.array([np.load(fn) for fn in fn_in_list])
    maps = maps.transpose((1,2,0))
    np.save(fn_out, maps)

def run_tiles(groupinfo):
    igroup, ngroup = groupinfo
    basedir = "/pscratch/sd/r/rosenber/so_sz/data/"
    templatemap_fn = f"{basedir}/masks/AdvACTSurveyMask_v7_galLatCut_S18_all1_pixell.fits" # Any car map of the right geometry as a template
    nside = 8  # Healpix NSIDE defining logical tiles
    field_size_deg = 14.8 # 1d square field size in degrees; None for auto
    nx = 4096
    dx_deg = field_size_deg / nx
    print(f"dx {dx_deg*60:02.2f} arcmin")
    projection_method='spline' # Method of projection 'sht' or 'spline'
    sht_lmax=10000 # lmax if doing sht projection; None for auto; not used if alm_fn is provided
    apod_pix = 50 # Number of pixels to apodize the edge of the map (for SHT interpolation)
    alm_fn = None#"/home/erik/jbca/Programs/szifi/data/so_maps/cmb+noise+sz_093GHz_cmbseed000_lat-063to+023deg_webskyps_beamfwhm_02p2arcmin_white008uK-arcmin_seed093_alm-lmax40000.npy" # .npy file containing alms for sht, None if unused
    order = 3  # Spline interpolation order; 3 for data, 0 for mask
    healpix_id_fn = f"{basedir}/cmb+noise+sz_car_f32/healpix_ids_sosimmask_nside{nside:03d}.txt" # Optional save ids of healpixels in our region
    verbosity = 2 # How much to print, 0,1,2
    imin, imax = 273-208, 274-208 # min, max index to calculate
    flip=True # Vertically flip tile; standard szifi behaviour

    do_galmask = True
    do_masks = True
    do_tiles = True
    
    if imax is None: imax = int(1e10)
    #field_shape = car.get_field_shape(field_size_deg, dx_deg, nside, min_buffer_deg) ## Set field shape from pixel size and field size
    field_shape = (nx, nx) ## Set field shape directly
    if projection_method == 'sht':
        projection_tag = f'{projection_method}{sht_lmax}'
    else:
        projection_tag = f'{projection_method}{order}'

    # Calculate groups for parallelization
    Nmax = len(get_healpix_ids(healpix_id_fn, map_filename=templatemap_fn, nside=nside))
    imin, imax = calculate_group_info(igroup, ngroup, imin, min(imax, Nmax))

    if do_galmask:
        ## Separately do a single galaxy mask with order 0 interpolation
        map_filename = f"{basedir}/masks/AdvACTSurveyMask_v7_galLatCut_S18_all1_pixell.fits"
        prefix_tile = f'{basedir}/so_tiles/cmb+noise+sz_galmask' # File path and name prefix for the saved fields
        get_cutout(map_filename, nside, dx_deg, field_shape, 'spline', None, False, True, None, prefix_tile, sht_lmax, alm_fn, 0, healpix_id_fn, verbosity, imin, imax, flip)

    freqs = ['027', '039', '093', '145', '225', '278']
    fn_info = [('027', '07p4', '071', '027'), ('039', '05p1', '036', '039'), ('093', '02p2', '008', '093'), ('145', '01p4', '010', '145'), ('225', '01p0', '022', '225'), ('278', '00p9', '054', '278')]

    prefix_mask = f'{basedir}/so_tiles/cmb+noise+sz' # File path and name prefix for the saved fields    
    for ii, freq in enumerate(freqs):
        map_filename = f"{basedir}/cmb+noise+sz_car_f32/cmb+noise+sz_%sGHz_cmbseed000_lat-063to+023deg_webskyps_beamfwhm_%sarcmin_white%suK-arcmin_seed%s.fits" % fn_info[ii]# Map to tile
        prefix_mask = f'{basedir}/so_tiles/cmb+noise+sz' # File path and name prefix for the saved fields
        prefix_tile = f'{basedir}/so_tiles/cmb+noise+sz_{freq}GHz-{projection_tag}' # File path and name prefix for the saved fields
        if ii == 0:
            get_cutout(map_filename, nside, dx_deg, field_shape, projection_method, apod_pix, do_masks, False, prefix_mask, prefix_tile, sht_lmax, alm_fn, order, healpix_id_fn, verbosity, imin, imax, flip) # Masks only the first time
        get_cutout(map_filename, nside, dx_deg, field_shape, projection_method, apod_pix, False, do_tiles, prefix_mask, prefix_tile, sht_lmax, alm_fn, order, healpix_id_fn, verbosity, imin, imax, flip)

def combine_freqs(groupinfo):
    """Combine individual frequency tiles into one. This should probably be done earlier but do here for now"""
    igroup, ngroup = groupinfo
    basedir = "/pscratch/sd/r/rosenber/so_sz/data"
    freqs = ['027', '039', '093', '145', '225', '278']
    projection_method = 'spline'
    order=3
    sht_lmax=10000
    healpix_id_fn = f"{basedir}/cmb+noise+sz_car_f32/healpix_ids_sosimmask_nside008.txt" # Optional save ids of healpixels in our region
    imin=273-208
    imax=274-208

    healpix_ids = get_healpix_ids(healpix_id_fn)
    Nmax = len(healpix_ids)
    imin, imax = calculate_group_info(igroup, ngroup, imin, min(imax, Nmax))

    projection_tag = f'{projection_method}{[order, sht_lmax][projection_method == "sht"]}'
    for hid in healpix_ids[imin:imax]:
        fn_tiles = [f'{basedir}/so_tiles/cmb+noise+sz_{freq}GHz-{projection_tag}_tile{hid}.npy' for freq in freqs]
        fn_out = f'{basedir}/so_tiles/cmb+noise+sz_{projection_tag}_tile{hid}.npy'
        collect_freqs(fn_tiles, fn_out)

def calculate_group_info(igroup, ngroup, imin, imax):
    Ntile = imax - imin
    w0, rr = Ntile//ngroup, Ntile % ngroup
    gwidth = [w0]*(ngroup-rr) + [w0+1]*rr
    start = np.concatenate(([0], np.cumsum(gwidth)[:-1])) + imin
    imin0 = start[igroup]
    imax0 = imin0 + gwidth[igroup]
    return imin0, imax0

# def main():
#     ngroup = 1 # This should be 1 for SHT (which can use all the cores) to prevent CPU oversubscription. Would be better to split those up too but more work needed
#     #fun = run_tiles
#     fun = combine_freqs
#     with Pool(ngroup) as pool:
#         pool.map(fun, ((ii, ngroup) for ii in range(ngroup)))

def main():
    combine_freqs((0,1))
    #run_tiles((0,1))

if __name__ == '__main__':
    main()
