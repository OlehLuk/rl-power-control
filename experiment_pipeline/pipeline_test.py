if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    from experiment_pipeline import run_ps_ql_experiments, best_combination_experiment
    import numpy as np

    n_episodes = 10
    n_steps = 20
    env_class = "experiment_results:JModelicaCSPSEnv"
    dym_env_class = "experiment_results:DymCSConfigurablePSEnv"
    _, episodes_lengths, exec_times, mse_rewards, eps_us, eps_ps, _, expl_perf, _ = \
        run_ps_ql_experiments(env_entry_point=dym_env_class,
                              n_experiments=2,
                              n_episodes=n_episodes,
                              max_n_steps=n_steps,
                              p_reff_period=400)

    print("Experiment length {} s".format(exec_times[0]))
    print(u"Avg episode length {} {} {}".format(episodes_lengths[0].mean(),
                                                chr(177),
                                                episodes_lengths[0].std()))
    print(u"All final mse {}".format(mse_rewards))

    print("Exploitation performances {} ".format(expl_perf))
    print(u"Avg exploitation performance {} {} {}".format(expl_perf[0].mean(),
                                                          chr(177),
                                                          expl_perf[0].std()))
    plt.plot(np.arange(1, n_episodes + 1, 1), mse_rewards[0])
    plt.show()

    for i in [0, 5, 9]:
        plt.plot(np.arange(0, n_steps + 1, 1), eps_ps[0][i])
        plt.plot(np.arange(0, n_steps + 1, 1), eps_us[0][i])
        plt.title("{}-th episode of training".format(i))
        plt.show()

    start = time.time()
    folder = "ampl_stoch_ps_exp_results"
    env_entry_point = "experiment_results:JModelicaCSPSEnv"
    # dym_env_class = "experiment_results:DymCSConfigurablePSEnv"
    stoch_env = "experiment_results:JMCSPSStochasticEnv"
    best_combination_experiment(folder, env_entry_point=stoch_env, t_s=[1, 5])

    # baseline_experiment(folder, [0.5, 1, 2, 5], env_entry_point)
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))

