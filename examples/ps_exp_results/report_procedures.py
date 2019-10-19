import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from IPython.core.pylabtools import figsize

FIRST_U = 0.8869
FIRST_P = 1.2

def report_experiment_summary(title, mses, perfs, sigs, labels, plot_interval_sigma=-1):
    styles = ['r--', 'y-.', 'k', 'b..']
    color = ['r', 'y', 'k', 'b']
    for i, mse in enumerate(mses):
        plt.plot(mse, styles[i], label=labels[i])
    if plot_interval_sigma >= 0:
        plt.hlines(perfs, xmin=mses[0].index.min(), xmax=mses[0].index.max(), colors=color[:len(mses)], 
                   label="Avg exploit performance")
    if plot_interval_sigma > 0:
        plt.hlines(np.array(perfs)+np.array(sigs), xmin=mses[0].index.min(), xmax=mses[0].index.max(), 
                   colors=color[:len(mses)], linestyles="dashed")
        plt.hlines(np.array(perfs)-np.array(sigs), xmin=mses[0].index.min(), xmax=mses[0].index.max(), 
                   colors=color[:len(mses)], linestyles="dashed")

    plt.title("Average smoothed last episode MSE vs training time. {}".format(title))
    plt.xlabel("# of episode")
    plt.ylabel("Average smoothed MSE")
    plt.legend();


def report_experiment(name, plot_perf_sigma=-1):
    figsize(9,6)
    df_ep_ls = pd.read_csv("./{}/episodes_lengths.csv".format(name), 
                           header=None)
    df_ex_ts = pd.read_csv("./{}/exec_times.csv".format(name), 
                           header=None)
    df_mses = pd.read_csv("./{}/mses.csv".format(name), 
                          header=None)
    df_expl_perf = pd.read_csv("./{}/exploit_performance.csv".format(name), 
                               header=None)
    df_avgsm_mses = plot_smoothed(df_mses, "PS Q-L", var_name="MSE", n_exp=5)
    perf, sigma = print_exec_time_summary(df_ep_ls, df_ex_ts, df_expl_perf)
    if plot_perf_sigma == 0:
        plt.hlines([perf], xmin=df_avgsm_mses.index.min(), xmax=df_avgsm_mses.index.max(),
                   colors=['k'], linestyles='dashed', label="Avg exploit. performance")
    elif plot_perf_sigma > 0:
        plt.hlines([perf, perf+plot_perf_sigma*sigma, perf-plot_perf_sigma*sigma], xmin=df_avgsm_mses.index.min(),
                   xmax=df_avgsm_mses.index.max(), colors=['k', 'k', 'k'], linestyles='dashed', 
               label="Avg exploit. performance with {}-sigma interval bounds".format(plot_perf_sigma))
    plt.legend()
    return df_avgsm_mses, perf, sigma

def report_const_experiment(name, old=False):
    figsize(15,15)
    subfolder_names = list(filter(lambda s: "k=" in s, os.listdir(name)))
    n_variations = len(subfolder_names)
    df_us = []
    df_ps = []
    df_mses = []
    for i, subf in enumerate(subfolder_names):
        df_us.append(pd.read_csv("{}/{}/us.csv".format(name, subf), header=None))
        df_ps.append(pd.read_csv("{}/{}/ps.csv".format(name, subf), header=None))
        df_mses.append(pd.read_csv("{}/{}/mses.csv".format(name, subf), header=None))

    for i, subf in enumerate(subfolder_names):
        plt.subplot(n_variations, 1, i+1)
        index = list(range(df_us[i].index.shape[0] + 1))
        if old:
            plt.plot(df_us[i].index.values, df_us[i].iloc[:, 0].values, 'red', label="Actual voltage")
            plt.plot(df_us[i].index.values, df_ps[i].iloc[:, 0].values, 'gray', label="Refference voltage")
        else:
            plt.plot(index, np.insert(df_us[i].iloc[:, 0].values, 0, FIRST_U), 'red', label="Actual voltage")
            plt.plot(index, np.insert(df_ps[i].iloc[:, 0].values, 0, FIRST_P), 'gray', label="Refference voltage")
        plt.title("Constant action {}. MSE={}".format(subf, df_mses[i].mean()))
        plt.legend()

def print_exec_time_summary(df_ep_ls, df_ex_ts, df_expl_perf):
    n_steps_in_exp = df_ep_ls.sum(axis=0).values
    t_per_step = df_ex_ts[0].values / n_steps_in_exp
    tot_ex_t = df_ex_ts.values.sum()
    tot_n_steps = n_steps_in_exp.sum()
    avg_per_step = tot_ex_t / tot_n_steps
    print("Time per simulation step in each experiment: {} s".format(t_per_step))
    print("Mean time per simulation step: {:.3f} s, std: {:.3f} s".format(t_per_step.mean(), t_per_step.std()))
    print("Total execution time: {:.3f} s for {} steps -> {:.3f} s per step". format(tot_ex_t, tot_n_steps, avg_per_step))
    
    performance = df_expl_perf.iloc[0,:].mean()
    sigma = df_expl_perf.iloc[0,:].std()
    print("\nAverage MSE in exploitation mode {:.4f} +- {:.4f}".format(performance, sigma))
    print("Average exploitatio performance of each agent: \n{}".format(df_expl_perf.mean(axis=0).values))
    return performance, sigma
    
def plot_smoothed(df, title_spec, var_name,  n_exp=5, n_smoothing=20):
    figsize(9,6)
    transformed = df.rolling(window=n_smoothing, min_periods=n_smoothing, axis=0).mean()[n_smoothing-1:]
    transformed['avg'] = transformed.mean(axis=1)
    plt.plot(transformed.index, transformed[[i for i in range(n_exp)]], 'gray', 
             alpha=0.5)
    plt.plot(transformed.index, transformed['avg'], 'red', label="MSE of episode")
    plt.title("Smoothed {} of episode vs training time. {}".format(var_name, title_spec))
    plt.xlabel("# of episode")
    plt.ylabel(var_name)
    plt.legend()
    print("Last episode performance: {:.2f}".format(transformed['avg'].iloc[-1]))
    return transformed['avg']


def plot_sys_behaviour(name, n_runs=5, steps=[0,-1], old=False):
    figsize(15,20)
    n_runs = 5
    df_us = [pd.read_csv("./{}/system_trajectory/us_run_{}.csv".format(name,i), header=None) for i in range(n_runs)]
    df_ps = [pd.read_csv("./{}/system_trajectory/ps_run_{}.csv".format(name,i), header=None) for i in range(n_runs)]
    df_mses = pd.read_csv("./{}/mses.csv".format(name), header=None)
    for i in range(n_runs):
        for j, step in enumerate(steps):
            index = list(range(df_us[i].index.shape[0] + 1))
            plt.subplot(n_runs, len(steps), len(steps)*i+j+1)
            if old:
                plt.plot(index, np.insert(df_us[i].iloc[:, step].values, 0, FIRST_U), 'red', label="Actual voltage")
                plt.plot(index,  np.insert(df_ps[i].iloc[:, step].values, 0, FIRST_P), 'gray', label="Refference voltage")
            else:
                plt.plot(df_us[i].index.values, df_us[i].iloc[:, step].values, 'red', label="Actual voltage")
                plt.plot(df_us[i].index.values,  df_ps[i].iloc[:, step].values, 'gray', label="Refference voltage")
            plt.title("Step #{}. MSE={}".format(step, df_mses.iloc[step,i]))
            plt.legend()
            
def plot_agent_behaviour(name, n_runs=5, steps=[0,-1]):
    figsize(15,20)
    n_runs = 5
    df_actions = [pd.read_csv("./{}/agent/action_run_{}.csv".format(name,i), header=None) for i in range(n_runs)]
    df_mses = pd.read_csv("./{}/mses.csv".format(name), header=None)
    for i in range(n_runs):
        for j, step in enumerate(steps):
            plt.subplot(n_runs, len(steps), len(steps)*i+j+1)
            plt.plot(df_actions[i].index.values, df_actions[i].iloc[:, step].values, 'o', label="Action performed (k)")
            plt.title("Step #{}. MSE={}".format(step, df_mses.iloc[step,i]))
            plt.legend()
            