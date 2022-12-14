import numpy as np
import pylab as pl
import cat

def subplot_binned_residuals(axs,catalogue_obs,catalogue_true,bins,redshift_bins,obs="q",set_xlabel=False,
ylabel=None,set_ylabel=False,legend=False,title=None,type="mean",label_legend=None,offset_factor=0.,bin_type="q",
color=None,bbox_to_anchor=[0.,0.],return_lgd=False,legend_loc="upper right",n_threshold=10.,find_label=False,
q_min=None,ylim=None,aspect=None):

    comparison = cat.catalogue_comparer(catalogue_true,catalogue_obs)
    catalogue_true,catalogue_obs = comparison.get_true_positives()

    q_true = catalogue_true.catalogue["q_opt"]

    if obs == "q":

        q_obs = catalogue_obs.catalogue["q_opt"]

        if find_label == False:

            delta = q_obs - q_true

        else:

            delta = q_obs - np.sqrt(q_true**2+3.)

    elif obs == "y0":

    #    if find_label == False:

        delta = (catalogue_obs.catalogue["y0"] - catalogue_true.catalogue["y0"])/catalogue_true.catalogue["y0"]

    #    elif find_label == True:
#
        #    sigma_opt = catalogue_obs.catalogue["y0"]/catalogue_obs.catalogue["q_opt"]
            #sigma_true = catalogue_true.catalogue["y0"]/catalogue_true.catalogue["q_opt"]
            #delta = np.sqrt(catalogue_true.catalogue["y0"]**2*sigma_opt**2/sigma_true**2+3.*sigma_opt**2)
            #delta = delta#/catalogue_true.catalogue["y0"]

    redshifts = catalogue_true.catalogue["z"]

    if q_min is not None:

        indices_min = np.where(q_true>q_min)
        q_true = q_true[indices_min]
        q_obs = q_obs[indices_min]
        delta = delta[indices_min]
        redshifts = redshifts[indices_min]

    print(len(catalogue_true.catalogue["z"]))
    print(len(catalogue_true.catalogue["y0"]))

    bins_centres = np.zeros(len(bins)-1)

    for i in range(0,len(bins_centres)):

        bins_centres[i] = 0.5*(bins[i]+bins[i+1])

    th_outlier = 4

    def get_binned(q_true_in,delta_q_in):

        binned_vec = np.zeros(len(bins_centres))
        n_vec = np.zeros(len(bins_centres))

        for k in range(0,len(bins_centres)):

            q_min = bins[k]
            q_max = bins[k+1]

            indices_bins = np.where((q_true_in > q_min) & (q_true_in < q_max) & (np.abs(delta_q_in)<th_outlier))

            n_vec[k] = len(indices_bins[0])

            if type == "mean":

                binned_vec[k] = np.mean(delta_q_in[indices_bins])

            elif type == "std":

                binned_vec[k] = np.std(delta_q_in[indices_bins])


        return binned_vec,n_vec

    def get_binned_std(q_true_in,delta_q_in):

        binned_vec = np.zeros(len(bins_centres))

        for k in range(0,len(bins_centres)):

            q_min = bins[k]
            q_max = bins[k+1]
            indices_bins = np.where((q_true_in > q_min) & (q_true_in < q_max) & (np.abs(delta_q_in)<th_outlier))

            if type == "mean":

                binned_vec[k] = np.std(delta_q_in[indices_bins])/np.sqrt(float(len(indices_bins[0])))

            elif type == "std":

                binned_vec[k] = np.std(delta_q_in[indices_bins])/np.sqrt(float(len(indices_bins[0])))

        return binned_vec

    def get_binned_bootstrap(q_true_in,delta_q_in,indices_boots):

        binned_matrix = np.zeros((len(indices_boots),len(bins)-1))

        for l in range(0,len(indices_boots)):

            q_true_in_b = q_true_in[indices_boots[l,:]]
            delta_q_in_b = delta_q_in[indices_boots[l,:]]

            #print(q_true_in_b,delta_q_in_b)

            binned_matrix[l,:],n_vec = get_binned(q_true_in_b,delta_q_in_b)

            #print(binned_matrix[l,:])

        return binned_matrix

    colors = ["tab:blue","tab:orange","tab:green","tab:red"]

    for i in range(0,len(redshift_bins)-1):

        indices_z = np.where((redshifts > redshift_bins[i]) & (redshifts < redshift_bins[i+1]))[0]

        #indices_boots_z = indices_boots[:,indices_z]

        bootnum = 1000
        indices_boots = np.random.randint(0,len(indices_z),size=(bootnum,len(indices_z)))

        delta_z = delta[indices_z]
        q_true_z = q_true[indices_z]
        redshifts_z = redshifts[indices_z]

        if bin_type == "q":

            delta_binned,n_vec = get_binned(q_true_z,delta_z)

            if type == "mean":

                delta_binned_std = get_binned_std(q_true_z,delta_z)

            elif type == "std":

                delta_matrix = get_binned_bootstrap(q_true_z,delta_z,indices_boots)
                delta_binned_std = np.std(delta_matrix,axis=0)
                #delta_binned_std = np.std(std_vec)

            offset_step = 0.5

        elif bin_type == "z":

            delta_binned,n_vec = get_binned(redshifts_z,delta_z)
        #    delta_matrix = get_binned_bootstrap(q_true_z,delta_z,indices_boots)

            if type == "mean":

                delta_binned_std = get_binned_std(redshifts_z,delta_z)

            elif type == "std":

                delta_matrix = get_binned_bootstrap(redshifts_z,delta_z,indices_boots)
                delta_binned_std = np.std(delta_matrix,axis=0)
                #print(delta_binned_std)
            #    delta_binned_std = np.std(std_vec)

            offset_step = 0.02

        offset = offset_step*offset_factor

        if label_legend is None:

            label_legend_n = str(redshift_bins[i]) + "$ < z < $" + str(redshift_bins[i+1])

        else:

            label_legend_n = label_legend

        if find_label == False:

            indices_plot = np.arange(0,len(bins_centres))

        else:

            indices_plot = np.arange(1,len(bins_centres))

        indices = np.where(n_vec >= n_threshold)


        #delta_binned[indices] = 0.
        #delta_binned_std[indices] = 0.

        indices_plot = np.intersect1d(indices,indices_plot)


        if color is not None:

            axs.errorbar(bins_centres[indices_plot]+offset,delta_binned[indices_plot],yerr=delta_binned_std[indices_plot],fmt='o',markersize='3',label=label_legend_n,color=color)

        else:

            axs.errorbar(bins_centres[indices_plot]+offset,delta_binned[indices_plot],yerr=delta_binned_std[indices_plot],fmt='o',markersize='3',label=label_legend_n,color=colors[i])

    if type == "mean":

        axs.plot(bins_centres,np.zeros(len(bins_centres)),color="k")

    elif type == "std":

        axs.plot(bins_centres,np.ones(len(bins_centres)),color="k")

    if title is not None:

        axs.set_title(title)

    if ylim is not None:

        axs.set_ylim(ylim)

    if legend == True and return_lgd == True:

        lgd = axs.legend(bbox_to_anchor=bbox_to_anchor,loc=legend_loc)

    elif  legend == True  and return_lgd == False:

        lgd = axs.legend(loc=legend_loc)

    if set_xlabel == True:

        if bin_type == "q":

            axs.set_xlabel(r"$\bar{q}_{\mathrm{t}}$")

        elif bin_type == "z":

            axs.set_xlabel(r"$z$")

    if set_ylabel == True:

        axs.set_ylabel(ylabel)

    if aspect is not None:

        axs.set(aspect=aspect)

    if return_lgd == True:

        ret = lgd

    else:

        ret = 0.

    return ret

def subplot_completeness(axs,catalogue_obs,catalogue_true,bins,redshift_bins,set_xlabel=False,
ylabel=None,set_ylabel=False,legend=False,title=None,q_th=5.,legend_text=None,opt_bias_label=False,
legend_loc_label=False,legend_loc="upper right",type="std"):

    #comparison = cat.catalogue_comparer(catalogue_true,catalogue_obs)
    #catalogue_true,catalogue_obs = comparison.get_true_positives()

    redshifts = catalogue_true.catalogue["z"]

    for i in range(0,len(redshift_bins)-1):

            #indices = np.where((redshifts < redshift_bins[i+1]) & (redshifts > redshift_bins[i]))[0]
            #catalogue_obs_z = cat.get_catalogue_indices(catalogue_obs,indices)
            #catalogue_true_z = cat.get_catalogue_indices(catalogue_true,indices)

            catalogue_obs_z = catalogue_obs
            catalogue_true_z = catalogue_true

            comparison = cat.catalogue_comparer(catalogue_true_z,catalogue_obs_z)
            bins_centres,completeness,completeness_err = comparison.get_completeness(bins,q_th=q_th)
            ret_completeness = cat.get_completeness_err(bins,catalogue_true,catalogue_obs,q_th=q_th,n_boots=1000,type=type)


            if legend_text is None:

                legendd = str(redshift_bins[i]) + "$ < z < $" + str(redshift_bins[i+1])

            else:

                legendd = legend_text

            indices_plot = np.where(completeness>0.)

            if type == "std":

                bins_centres,completenss_err = ret_completeness
                axs.errorbar(bins_centres[indices_plot],completeness[indices_plot],yerr=completeness_err[indices_plot],fmt='o',markersize='3',label=legendd)

            elif type == "quantile":

                bins_centres,completeness_low,completeness_high = ret_completeness

                bins_centres = bins_centres[indices_plot]
                completeness = completeness[indices_plot]
                completeness_err = np.array(list(zip(completeness-completeness_low[indices_plot],completeness_high[indices_plot]-completeness))).T

                axs.errorbar(bins_centres,completeness,yerr=completeness_err,fmt='o',markersize='3',label=legendd)


    bins_erf = np.linspace(bins[0],bins[-1],100)
    erf_completeness_fixed = cat.get_erf_completeness(bins_erf,q_th=q_th,opt_bias=opt_bias_label)
    axs.plot(bins_erf,erf_completeness_fixed)

    if title is not None:

        axs.set_title(title)

    if legend == True:

        if legend_loc_label == False:

            axs.legend()

        else:

            axs.legend(loc=legend_loc)

    if set_xlabel == True:

        axs.set_xlabel(r"$\bar{q}_{\mathrm{t}}$")

    if set_ylabel == True:

        axs.set_ylabel(ylabel)
