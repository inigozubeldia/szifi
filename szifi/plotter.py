import numpy as np
import pylab as pl
import cat

def subplot_binned_residuals(axs,catalogue_obs,catalogue_true,bins,redshift_bins,obs="q",set_xlabel=False,
ylabel=None,set_ylabel=False,legend=False,title=None,type="mean"):

    comparison = cat.catalogue_comparer(catalogue_true,catalogue_obs)
    catalogue_true,catalogue_obs = comparison.get_true_positives()

    q_true = catalogue_true.catalogue["q_opt"]

    if obs == "q":

        q_obs = catalogue_obs.catalogue["q_opt"]
        delta = q_obs - q_true

    elif obs == "y0":

        delta = (catalogue_obs.catalogue["y0"] - catalogue_true.catalogue["y0"])/catalogue_true.catalogue["y0"]

    redshifts = catalogue_true.catalogue["z"]

    bins_centres = np.zeros(len(bins)-1)

    for i in range(0,len(bins_centres)):

        bins_centres[i] = 0.5*(bins[i]+bins[i+1])

    th_outlier = 1000

    def get_binned(q_true_in,delta_q_in):

        binned_vec = np.zeros(len(bins_centres))

        for k in range(0,len(bins_centres)):

            q_min = bins[k]
            q_max = bins[k+1]
            indices_bins = np.where((q_true_in > q_min) & (q_true_in < q_max) & (np.abs(delta_q_in)<th_outlier))

            if type == "mean":

                binned_vec[k] = np.mean(delta_q_in[indices_bins])

            elif type == "std":

                binned_vec[k] = np.std(delta_q_in[indices_bins])

        return binned_vec

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

            binned_matrix[l,:] = get_binned(q_true_in[indices_boots[l,:]],delta_q_in[indices_boots[l,:]])

        return binned_matrix

    for i in range(0,len(redshift_bins)-1):

        indices_z = np.where((redshifts > redshift_bins[i]) & (redshifts < redshift_bins[i+1]))[0]

        #indices_boots_z = indices_boots[:,indices_z]

        bootnum = 1000
        indices_boots = np.random.randint(0,len(indices_z),size=(bootnum,len(indices_z)))

        delta_z = delta[indices_z]
        q_true_z = q_true[indices_z]
        delta_binned = get_binned(q_true_z,delta_z)
    #    delta_matrix = get_binned_bootstrap(q_true_z,delta_z,indices_boots)
        delta_binned_std = get_binned_std(q_true_z,delta_z)

        axs.errorbar(bins_centres,delta_binned,yerr=delta_binned_std,fmt='o',markersize='3',label=str(redshift_bins[i]) + "$ < z < $" + str(redshift_bins[i+1]))

    if type == "mean":

        axs.plot(bins_centres,np.zeros(len(bins_centres)))

    elif type == "std":

        axs.plot(bins_centres,np.ones(len(bins_centres)))

    if title is not None:

        axs.set_title(title)

    if legend == True:

        axs.legend()

    if set_xlabel == True:

        axs.set_xlabel(r"$\bar{q}_{\mathrm{t}}$")

    if set_ylabel == True:

        axs.set_ylabel(ylabel)

def subplot_completeness(axs,catalogue_obs,catalogue_true,bins,redshift_bins,set_xlabel=False,
ylabel=None,set_ylabel=False,legend=False,title=None,q_th=5.):

    comparison = cat.catalogue_comparer(catalogue_true,catalogue_obs)
    catalogue_true,catalogue_obs = comparison.get_true_positives()

    redshifts = catalogue_true.catalogue["z"]

    for i in range(0,len(redshift_bins)-1):

            indices = np.where((redshifts < redshift_bins[i+1]) & (redshifts > redshift_bins[i]))[0]
            catalogue_obs_z = cat.get_catalogue_indices(catalogue_obs,indices)
            catalogue_true_z = cat.get_catalogue_indices(catalogue_true,indices)

            comparison = cat.catalogue_comparer(catalogue_true_z,catalogue_obs_z)
            bins_centres,completeness,completeness_err = comparison.get_completeness(bins,q_th=q_th)
            bins_centres,completeness_err = cat.get_completeness_err(bins,catalogue_true,catalogue_obs,q_th=q_th,n_boots=1000)

            axs.errorbar(bins_centres,completeness,yerr=completeness_err,fmt='o',markersize='3',label=str(redshift_bins[i]) + "$ < z < $" + str(redshift_bins[i+1]))

    bins_erf = np.linspace(bins[0],bins[-1],100)
    erf_completeness_fixed = cat.get_erf_completeness(bins_erf,q_th=q_th,opt_bias=False)
    axs.plot(bins_erf,erf_completeness_fixed)

    if title is not None:

        axs.set_title(title)

    if legend == True:

        axs.legend()

    if set_xlabel == True:

        axs.set_xlabel(r"$\bar{q}_{\mathrm{t}}$")

    if set_ylabel == True:

        axs.set_ylabel(ylabel)
