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

def get_matched_filter_real(rec,tem,noi):

    map1 = rec*tem/noi
    map2 = tem**2/noi

    sum_1 = np.sum(map1)
    sum_2 = np.sum(map2)

    normalisation_estimate = sum_1/sum_2
    normalisation_variance = 1./sum_2

    return (normalisation_estimate,normalisation_variance)

def get_matched_filter(rec,tem,noi,pix,theta_max_arcmin=None,theta_min_arcmin=None,
lmin=0,lmax=1e5):

    rec2 = np.copy(rec)
    tem2 = np.copy(tem)

    if theta_max_arcmin is None:

        theta_max_arcmin = 1e5

    if theta_min_arcmin is None:

        theta_min_arcmin = 0.

    theta_max = theta_max_arcmin/60./180.*np.pi
    theta_min = theta_min_arcmin/60./180.*np.pi
    distance_map = maps.rmap(pix).get_distance_map_wrt_centre()
    indices = np.where(distance_map > theta_max)
    rec2[indices] = 0.
    tem2[indices] = 0.
    indices = np.where(distance_map < theta_min)
    rec2[indices] = 0.
    tem2[indices] = 0.

    if lmin > 0 or lmax < 1e5:

        indices = np.where((maps.rmap(pix).get_ell() < lmin) |  (maps.rmap(pix).get_ell() > lmax))
        noi[indices] = 0.

    norm_estimate,norm_variance = filter_sum(maps.get_fft(rec2,pix),maps.get_fft(tem2,pix),noi)

    return (norm_estimate,norm_variance)



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

def get_cluster_mask_one(pix,catalogue,mask_radius):

    x_est,y_est = maps.get_theta_misc([catalogue.catalogue["theta_x"],catalogue.catalogue["theta_y"]],pix)
    source_coords = np.zeros((1,2))
    source_coords[0,0] = x_est
    source_coords[0,1] = y_est
    mask_cluster = maps.ps_mask(pix,1,catalogue.catalogue["theta_500"]*mask_radius).get_mask_map(source_coords=source_coords)

    return mask_cluster

def select_true_catalogue(true_catalogue,theta_500_vec,sigma_vec,n_clusters=10):

    if len(true_catalogue.catalogue["lon"]) > n_clusters:

        q_true = true_catalogue.catalogue["y0"]/np.interp(true_catalogue.catalogue["theta_500"],theta_500_vec,sigma_vec)
        indices = np.argpartition(q_true,-n_clusters)[-n_clusters:]
        ret = cat.get_catalogue_indices(true_catalogue,indices)

    else:

        ret = true_catalogue

    return ret
