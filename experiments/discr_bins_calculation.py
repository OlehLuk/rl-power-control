import pandas as pd
import numpy as np


def load_baselines_to_array(baseline_folder, ks=(0.5, 1, 2, 3, 4, 5, 6, 7)):
    ars = [pd.read_csv("{}/k={}/us.csv".format(baseline_folder, k), header=None).values.flatten() for k in ks]
    return np.concatenate(ars)


def load_experiment_to_array(experiment_folder, n_episodes, n_runs=5):
    ars = [pd.read_csv("{}/system_trajectory/us_run_{}.csv".format(
        experiment_folder, i), header=None).iloc[:, :n_episodes].values.flatten() for i in range(n_runs)]
    return np.concatenate(ars)


def calc_quantile_bins(us, n=10):
    qs = np.linspace(0, 1, n+1)
    return np.quantile(us, qs)


def calc_opt_hist_bins(us):
    return np.histogram_bin_edges(us, bins='auto')


def load_bins(filename):
    return pd.read_csv(filename, header=None)[0].values[1:-1]


def bins_with_p_reff(p_reff, min_p, max_p, n_bins):
    # step = (max_p - min_p) / n_bins
    # assuming, that with real data p_reff is not an edge of perfect bins
    step = (max_p - min_p) / (n_bins - 1)
    lower = np.arange(p_reff, min_p, -step)
    upper = np.arange(p_reff, max_p, step)
    result = np.concatenate([[lower[-1]-step], lower[::-1], upper[1:], [upper[-1] + step]])
    return result

if __name__ == "__main__":
    bins_folder = "results/dicretization_bins"

    # baseline_folder = "results/ps_stoch/constant_action_variation_20-10-2019_19-55"
    # result_filename = "baseline12_quantile_10bins.csv"
    # us = load_baselines_to_array(baseline_folder)
    # bins = calc_quantile_bins(us, n=10)
    # np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
    #           X=bins,
    #           delimiter=",",
    #           fmt="%.8f")

    # result_filename = "baseline12_quantile_25bins.csv"
    # bins = calc_quantile_bins(us, n=25)
    # np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
    #           X=bins,
    #           delimiter=",",
    #           fmt="%.8f")

    # result_filename = "baseline12_hist_bins.csv"
    # bins = calc_opt_hist_bins(us)
    # np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
    #           X=bins,
    #           delimiter=",",
    #           fmt="%.8f")

    # experiment_folder = "results/ps_stoch/best_parameters_combination_19-09-2019_13-33/time_step=1"
    # us = load_experiment_to_array(experiment_folder, n_episodes=32)


    """
    result_filename = "ql2_32ep_hist_bins.csv"
    bins = calc_opt_hist_bins(us)
    print(len(bins))
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")
    print(len(load_bins("{}/{}".format(bins_folder, result_filename))))

    result_filename = "ql2_32ep_quantile_10bins.csv"
    bins = calc_quantile_bins(us, n=10)
    print(bins)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")
    print(len(load_bins("{}/{}".format(bins_folder, result_filename))))

    result_filename = "ql2_32ep_quantile_25bins.csv"
    bins = calc_quantile_bins(us, n=25)
    print(bins)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")
    print(len(load_bins("{}/{}".format(bins_folder, result_filename))))

    
    baseline_folder = "results/ps_stoch/constant_action_variation_20-10-2019_19-55"
    result_filename = "baseline_p_ref_10bins.csv"
    us = load_baselines_to_array(baseline_folder)
    print("max={:.6f}, min={:.6f}".format(us.max(), us.min()))
    bins = bins_with_p_reff(1.2, us.min(), us.max(), 10)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")

    experiment_folder = "results/ps_stoch/best_parameters_combination_19-09-2019_13-33/time_step=1"
    us = load_experiment_to_array(experiment_folder, n_episodes=32)
    result_filename = "ql12_32ep_p_ref_10bins.csv"
    print("max={:.6f}, min={:.6f}".format(us.max(), us.min()))
    bins = bins_with_p_reff(1.2, us.min(), us.max(), 10)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")
    """

    baseline_folder = "results/ps_stoch/skip_transition_experiment/baseline_skip_transition175/p_reff=1.2/"
    result_filename = "bas_skip175_p_ref_30bins.csv"
    us = load_baselines_to_array(baseline_folder)
    print("max={:.6f}, min={:.6f}".format(us.max(), us.min()))
    bins = bins_with_p_reff(1.2, us.min(), us.max(), 30)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")
    """
    result_filename = "bas_skip175_quantile_10bins.csv"
    bins = calc_quantile_bins(us)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")

    result_filename = "bas_skip175_hist_bins.csv"
    bins = calc_opt_hist_bins(us)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")

    experiment_folder = \
        "results/ps_stoch/skip_transition_experiment/best_params_skip_transition175/p_reff=1.2/time_step=1"
    us = load_experiment_to_array(experiment_folder, n_episodes=32)

    result_filename = "ql12_32ep_skip175_p_ref_10bins.csv"
    print("max={:.6f}, min={:.6f}".format(us.max(), us.min()))
    bins = bins_with_p_reff(1.2, us.min(), us.max(), 10)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")

    result_filename = "ql12_32ep_skip175_quantile_10bins.csv"
    bins = calc_quantile_bins(us)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")

    result_filename = "ql12_32ep_skip175_histbins.csv"
    bins = calc_opt_hist_bins(us)
    np.savetxt(fname="{}/{}".format(bins_folder, result_filename),
               X=bins,
               delimiter=",",
               fmt="%.8f")
    """

# [0.7539 0.9522 1.0172 1.0629 1.0828 1.1455 1.2086 1.2711 1.3327 1.3619 1.6262]