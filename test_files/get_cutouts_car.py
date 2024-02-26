from pixell import enmap
import numpy as np
import szifi.car as car

def get_cutout(map_filename, nside, dx_deg, field_shape, projection_method, apod_pix, do_masks, do_tiles, prefix_mask, prefix_tile, sht_lmax=None, alm_fn=None, order=3, healpix_id_fn=None, verbosity=1, imin=0, imax=int(1e10)):
    if do_masks:
        map_geometry = enmap.read_fits_geometry(map_filename)
        tiles_car = car.make_healpix_tiles_car(nside, map_geometry)  ## Need to redo to project mask directly from healpix to desired field
        healpix_ids = np.unique(tiles_car)
        del tiles_car
        
        if healpix_id_fn is not None: np.savetxt(healpix_id_fn, healpix_ids)
        car.make_all_masks(nside, dx_deg, field_shape, healpix_ids, prefix_mask, verbosity, imin, imax)

    if do_tiles:
        if healpix_id_fn is not None:
            healpix_ids = np.loadtxt(healpix_id_fn).astype('int64')
        else:
            healpix_ids = None
        imap = enmap.read_map(map_filename)
        if (apod_pix is not None) and apod_pix > 0:
            imap = enmap.apod(imap, apod_pix) # something more advanced needed if there is a non-trivial footprint mask
        car.make_all_tiles(imap, nside, dx_deg, field_shape, healpix_ids, projection_method, sht_lmax, alm_fn, order, prefix_tile, verbosity, imin, imax)
    
def main():
    basedir = "/home/erik/jbca/Programs/szifi/data"
    nside = 8  # Healpix NSIDE defining logical tiles
    field_size_deg = 14.8 # 1d square field size in degrees; None for auto
    nx = 4096
    dx_deg = field_size_deg / nx
    print(f"dx {dx_deg*60:02.2f} arcmin")
    projection_method='spline' # Method of projection 'sht' or 'spline'
    sht_lmax=None # lmax if doing sht projection; None for auto; not used if alm_fn is provided
    alm_fn = None#"/home/erik/jbca/Programs/szifi/data/so_maps/cmb+noise+sz_093GHz_cmbseed000_lat-063to+023deg_webskyps_beamfwhm_02p2arcmin_white008uK-arcmin_seed093_alm-lmax40000.npy" # .npy file containing alms for sht, None if unused
    order = 3  # Spline interpolation order
    healpix_id_fn = f"{basedir}/so_maps/healpix_ids_sosimmask_nside{nside:03d}.txt" # Optional save ids of healpixels in our region
    verbosity = 2 # How much to print, 0,1,2
    imin, imax = 0, 3 # min, max index to calculate
    do_masks = False
    do_tiles = True
    apod_pix = None # Number of pixels to apodize the edge of the map for SHT interpolation
    
    if imax is None: imax = int(1e10)
    #field_shape = car.get_field_shape(field_size_deg, dx_deg, nside, min_buffer_deg) ## Set field shape from pixel size and field size
    field_shape = (nx, nx) ## Set field shape directly


    #map_filename = "/home/erik/jbca/so_sz/data/masks/AdvACTSurveyMask_v7_galLatCut_S18_all1_pixell.fits"
    #prefix_tile = f'/home/erik/jbca/Programs/szifi/data/so_tiles/cmb+noise+sz_093GHz-galmask' # File path and name prefix for the saved fields
    freqs = ['027', '039', '093', '145', '225', '278']
    fn_info = [('027', '07p4', '071', '027'), ('039', '05p1', '036', '039'), ('093', '02p2', '008', '093'), ('143', '01p4', '010', '143'), ('225', '01p0', '022', '225'), ('278', '00p9', '054', '278')]
    
    for ii, freq in enumerate(freqs):
        map_filename = f"{basedir}/so_maps/cmb+noise+sz_%sGHz_cmbseed000_lat-063to+023deg_webskyps_beamfwhm_%sarcmin_white%suK-arcmin_seed%s.fits" % fn_info[ii]# Map to tile
        prefix_mask = f'{basedir}/so_tiles/cmb+noise+sz_{freq}GHz' # File path and name prefix for the saved fields
        #prefix_tile = f'{basedir}/so_tiles/cmb+noise+sz_093GHz-{projection_method}{sht_lmax}' # File path and name prefix for the saved fields
        prefix_tile = f'/home/erik/jbca/Programs/szifi/data/so_tiles/cmb+noise+sz_{freq}GHz-{projection_method}' # File path and name prefix for the saved fields    
        get_cutout(map_filename, nside, dx_deg, field_shape, projection_method, apod_pix, do_masks, do_tiles, prefix_mask, prefix_tile, sht_lmax, alm_fn, order, healpix_id_fn, verbosity, imin, imax)

if __name__ == '__main__':
    main()
