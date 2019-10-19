import logging
import numpy as np
import os
from examples.ps_q_learner import run_ps_ql_experiments, run_ps_const_agent_experiments
from datetime import datetime
import pickle as pkl


def run_baseline_experiment_with_files(base_folder,
                                       const_action,
                                       n_episodes,
                                       max_n_steps,
                                       p_reff,
                                       time_step,
                                       log_level,
                                       env_entry_point,
                                       **kwargs):
    episode_lengths, episodes_mse_reward, episodes_us, episodes_ps = run_ps_const_agent_experiments(
        env_entry_point=env_entry_point,
        const_action=const_action,
        n_episodes=n_episodes,
        max_n_steps=max_n_steps,
        p_reff=p_reff,
        time_step=time_step,
        log_level=log_level,
        **kwargs)

    np.savetxt(fname="{}/episodes_lengths.csv".format(base_folder),
               X=np.transpose(episode_lengths),
               delimiter=",",
               fmt="%d")
    np.savetxt(fname="{}/mses.csv".format(base_folder),
               X=np.transpose(episodes_mse_reward),
               delimiter=",",
               fmt="%.4f")
    np.savetxt(fname="{}/us.csv".format(base_folder),
               X=np.transpose(episodes_us),
               delimiter=",",
               fmt="%.4f")

    np.savetxt(fname="{}/ps.csv".format(base_folder),
               X=np.transpose(episodes_ps),
               delimiter=",",
               fmt="%.4f")


def run_experiment_with_result_files(base_folder,
                                     n_experiments,
                                     n_episodes,
                                     visualize,
                                     max_n_steps,
                                     time_step,
                                     p_reff,
                                     rand_qtab,
                                     learning_rate,
                                     discount_factor,
                                     exploration_rate,
                                     exploration_decay_rate,
                                     k_s,
                                     log_level,
                                     env_entry_point,
                                     compute_reward=None,
                                     **kwargs):
    """
    Runs experiments with the given configuration and writes episodes length of all experiment as one file
    and execution times of experiments as another.
    File names are composed from numerical experiment parameters
    in the same order as in function definition.
    Episodes length are written as 2d-array of shape (n_episodes, n_experiments):
    i-th row - i-th episode, j-th column - j-th experiment.

    Execution times are written as 1d-array of shape (n_experiments, ): j-th element - j-th experiment


    :param base_folder: folder for experiment result files
    :return: None
    """

    agents, episodes_lengths, exec_times, mse_rewards_s, eps_us, eps_ps, eps_acs, expl_perfs, expl_actns = \
        run_ps_ql_experiments(
            env_entry_point=env_entry_point,
            n_experiments=n_experiments,
            n_episodes=n_episodes,
            visualize=visualize,
            max_n_steps=max_n_steps,
            p_reff=p_reff,
            time_step=time_step,
            rand_qtab=rand_qtab,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay_rate=exploration_decay_rate,
            k_s=k_s,
            log_level=log_level,
            compute_reward=compute_reward,
            **kwargs)

    np.savetxt(fname="{}/episodes_lengths.csv".format(base_folder),
               X=np.transpose(episodes_lengths),
               delimiter=",",
               fmt="%d")
    np.savetxt(fname="{}/exec_times.csv".format(base_folder),
               X=np.array(exec_times),
               delimiter=",",
               fmt="%.4f")
    np.savetxt(fname="{}/mses.csv".format(base_folder),
               X=np.transpose(mse_rewards_s),
               delimiter=",",
               fmt="%.4f")

    np.savetxt(fname="{}/exploit_performance.csv".format(base_folder),
               X=np.transpose(expl_perfs),
               delimiter=",",
               fmt="%.4f")

    sys_trajectory_folder = "{}/system_trajectory".format(base_folder)
    agent_folder = "{}/agent".format(base_folder)

    os.mkdir(sys_trajectory_folder)
    os.mkdir(agent_folder)

    for i in range(n_experiments):
        np.savetxt(fname="{}/us_run_{}.csv".format(sys_trajectory_folder, i),
                   X=np.transpose(eps_us[i]),
                   delimiter=",",
                   fmt="%.4f")
        np.savetxt(fname="{}/ps_run_{}.csv".format(sys_trajectory_folder, i),
                   X=np.transpose(eps_ps[i]),
                   delimiter=",",
                   fmt="%.4f")

        np.savetxt(fname="{}/trained_qtable_run_{}.csv".format(agent_folder, i),
                   X=np.array(agents[i].qtable),
                   delimiter=",",
                   fmt="%.4f")

        with open("{}/trained_agent_run_{}.pkl".format(agent_folder, i), "wb") as f:
            pkl.dump(agents[i], f)

        np.savetxt(fname="{}/action_run_{}.csv".format(agent_folder, i),
                   X=np.transpose(eps_acs[i]),
                   delimiter=",",
                   fmt="%.4f")

        np.savetxt(fname="{}/exploit_action_run_{}.csv".format(agent_folder, i),
                   X=np.transpose(expl_actns[i]),
                   delimiter=",",
                   fmt="%.4f")

    # np.savetxt(fname=experiment_file_name_prefix + "us.csv",
    #           X=np.array(eps_us),
    #           delimiter=",",
    #           fmt="%.4f")


def prepare_experiment(base_folder, experiment_name):
    now = datetime.now()
    experiment_folder = "{}/{}_{}".format(base_folder, experiment_name, now.strftime("%d-%m-%Y_%H:%M"))

    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    else:
        print("Terminating: possible override of previous experiment results is detected.")
        return None

    return experiment_folder


def baseline_experiment(base_folder, const_actions, env_entry_point):
    experiment_name = "constant_action_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - {}".format(i, k_s) for i, k_s in enumerate(const_actions)])
        f.write("""{}

Changed constant actions in: 
{}

Other experiment parameters were fixed:
    n_episodes=10,
        max_n_steps=20,
        time_step=1,
        p_reff=1.2,
        const_action=a,
        log_level=logging.INFO
            """.format(experiment_name, param_i_correspondence))

    for a in const_actions:
        subfolder = "{}/k={}".format(experiment_folder, a)
        os.mkdir(subfolder)
        run_baseline_experiment_with_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_episodes=10,
            max_n_steps=200,
            time_step=1,
            p_reff=1.2,
            const_action=a,
            log_level=logging.INFO,
            p_reff_amplitude=0,
            p_reff_period=200
        )


def ks_experiment(base_folder, ks, env_entry_point):
    experiment_name = "action_space(k_s)_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - {}".format(i, k_s) for i, k_s in enumerate(ks)])
        f.write("""{}

Changed k_s in: 
{}

Other experiment parameters were fixed:
    n_experiments=5,
    n_episodes=100,
    visualize=False,
    max_n_steps=200,
    time_step=1,
    p_reff=1.3,
    rand_qtab=False,
    learning_rate=0.5,
    discount_factor=0.6,
    exploration_rate=0.5,
    exploration_decay_rate=0.9,
    log_level=logging.INFO
        """.format(experiment_name, param_i_correspondence))

    for k_s in ks:
        subfolder = "{}/k_s={}".format(experiment_folder, k_s)
        os.mkdir(subfolder)
        run_experiment_with_result_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_experiments=5,
            n_episodes=100,
            visualize=False,
            max_n_steps=200,
            time_step=1,
            p_reff=1.3,
            rand_qtab=False,
            learning_rate=0.5,
            discount_factor=0.6,
            exploration_rate=0.5,
            exploration_decay_rate=0.9,
            k_s=k_s,
            log_level=logging.INFO
        )


def exploration_experiment(base_folder, explor_params, env_entry_point):
    experiment_name = "exploration(rate_and_discount)_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - exploration rate = {}, exploration rate decay = {}".format(
            i, params[0], params[1]) for i, params in enumerate(explor_params)])
        f.write("""{}

Changed exploration parameters in: 
{}

Other experiment parameters were fixed:
    n_experiments=5,
    n_episodes=100,
    visualize=False,
    max_n_steps=200,
    time_step=1,
    p_reff=1.3,
    rand_qtab=False,
    learning_rate=0.5,
    discount_factor=0.6,
    k_s=[0.1, 0.5, 1, 2, 7],    
    log_level=logging.INFO
        """.format(experiment_name, param_i_correspondence))

    for expl_rate, expl_decay in explor_params:
        subfolder = "{}/expl_rate={}_decay={}".format(experiment_folder, expl_rate, expl_decay)
        os.mkdir(subfolder)
        run_experiment_with_result_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_experiments=5,
            n_episodes=100,
            visualize=False,
            max_n_steps=200,
            time_step=1,
            p_reff=1.3,
            rand_qtab=False,
            learning_rate=0.5,
            discount_factor=0.6,
            exploration_rate=expl_rate,
            exploration_decay_rate=expl_decay,
            k_s=[0.1, 0.5, 1, 2, 7],
            log_level=logging.INFO
        )


def timestep_experiment(base_folder, t_s, env_entry_point):
    experiment_name = "time_step_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - time_step = {}".format(
            i, t) for i, t in enumerate(t_s)])
        f.write("""{}

Changed time step in: 
{}

Other experiment parameters were fixed:
    n_experiments=5,
    n_episodes=100,
    visualize=False,
    max_n_steps=200,
    p_reff=1.3,
    rand_qtab=False,
    learning_rate=0.5,
    discount_factor=0.6,
    k_s=[0.1, 0.5, 1, 2, 7],
    exploration_rate=0.5,
    exploration_decay_rate=0.9,   
    log_level=logging.INFO
        """.format(experiment_name, param_i_correspondence))

    for t in t_s:
        subfolder = "{}/time_step={}".format(experiment_folder, t)
        os.mkdir(subfolder)
        max_n_steps = 200 // t
        run_experiment_with_result_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_experiments=5,
            n_episodes=100,
            visualize=False,
            max_n_steps=max_n_steps,
            time_step=t,
            p_reff=1.3,
            rand_qtab=False,
            learning_rate=0.5,
            discount_factor=0.6,
            exploration_rate=0.5,
            exploration_decay_rate=0.9,
            k_s=[0.1, 0.5, 1, 2, 7],
            log_level=logging.INFO
        )


def learning_rate_experiment(base_folder, lr_s, env_entry_point):
    experiment_name = "learning_rate_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - learning_rate = {}".format(
            i, lr) for i, lr in enumerate(lr_s)])
        f.write("""{}

Learning rate changed in: 
{}

Other experiment parameters were fixed:
    n_experiments=5,
    n_episodes=100,
    visualize=False,
    time_step=1,
    max_n_steps=200,
    p_reff=1.3,
    rand_qtab=False,
    discount_factor=0.6,
    k_s=[0.1, 0.5, 1, 2, 7],
    exploration_rate=0.5,
    exploration_decay_rate=0.9,   
    log_level=logging.INFO
        """.format(experiment_name, param_i_correspondence))

    for lr in lr_s:
        subfolder = "{}/learning_rate={}".format(experiment_folder, lr)
        os.mkdir(subfolder)
        run_experiment_with_result_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_experiments=5,
            n_episodes=100,
            visualize=False,
            max_n_steps=200,
            time_step=1,
            p_reff=1.3,
            rand_qtab=False,
            learning_rate=lr,
            discount_factor=0.6,
            exploration_rate=0.5,
            exploration_decay_rate=0.9,
            k_s=[0.1, 0.5, 1, 2, 7],
            log_level=logging.INFO
        )


def discount_factor_experiment(base_folder, df_s, env_entry_point):
    experiment_name = "discount_factor_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - discount_factor = {}".format(
            i, df) for i, df in enumerate(df_s)])
        f.write("""{}

Discount rate changed in: 
{}

Other experiment parameters were fixed:
    n_experiments=5,
    n_episodes=100,
    visualize=False,
    time_step=1,
    max_n_steps=200,
    p_reff=1.3,
    rand_qtab=False,
    learning_rate=0.5,
    k_s=[0.1, 0.5, 1, 2, 7],
    exploration_rate=0.5,
    exploration_decay_rate=0.9,   
    log_level=logging.INFO
        """.format(experiment_name, param_i_correspondence))

    for df in df_s:
        subfolder = "{}/discount_factor={}".format(experiment_folder, df)
        os.mkdir(subfolder)
        run_experiment_with_result_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_experiments=5,
            n_episodes=100,
            visualize=False,
            max_n_steps=200,
            time_step=1,
            p_reff=1.3,
            rand_qtab=False,
            learning_rate=0.5,
            discount_factor=df,
            exploration_rate=0.5,
            exploration_decay_rate=0.9,
            k_s=[0.1, 0.5, 1, 2, 7],
            log_level=logging.INFO
        )


def reward_experiment(base_folder, env_entry_point, compute_reward_s):
    experiment_name = "reward_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)

    param_i_correspondence = "\n".join(["{} - reward = {}".format(
        i, comp_rew[0]) for i, comp_rew in enumerate(compute_reward_s)])

    with open(description_file, "w") as f:
        f.write("""{}
        
Reward changed in:
{}

Experiment parameters were fixed:
    n_experiments=5,
    n_episodes=100,
    visualize=False,
    time_step=5,
    max_n_steps=40,
    p_reff=1.3,
    rand_qtab=False,
    learning_rate=0.5,
    discount_factor=0.6,
    k_s=[0.1, 0.5, 1, 2, 7],
    exploration_rate=0.5,
    exploration_decay_rate=0.9,   
    log_level=logging.INFO
        """.format(experiment_name, param_i_correspondence))

    for comp_rew in compute_reward_s:
        name, func = comp_rew
        subfolder = "{}/reward-{}".format(experiment_folder, name)
        os.mkdir(subfolder)
        run_experiment_with_result_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_experiments=5,
            n_episodes=100,
            visualize=False,
            max_n_steps=200,
            time_step=1,
            p_reff=1.3,
            rand_qtab=False,
            learning_rate=0.5,
            discount_factor=0.6,
            exploration_rate=0.5,
            exploration_decay_rate=0.9,
            k_s=[0.1, 0.5, 1, 2, 7],
            log_level=logging.INFO,
            compute_reward=func
        )


def validation_experiment(base_folder, env_entry_point):
    experiment_name = "setup_validation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        f.write("""{}

Experiment parameters were fixed:
    n_experiments=5,
    n_episodes=100,
    visualize=False,
    time_step=5,
    max_n_steps=40,
    p_reff=1.3,
    rand_qtab=False,
    learning_rate=0.5,
    discount_factor=0.6,
    k_s=[0.1, 0.5, 1, 2, 7],
    exploration_rate=0.5,
    exploration_decay_rate=0.9,   
    log_level=logging.INFO
        """.format(experiment_name))

    run_experiment_with_result_files(
        env_entry_point=env_entry_point,
        base_folder=experiment_folder,
        n_experiments=5,
        n_episodes=100,
        visualize=False,
        max_n_steps=200,
        time_step=1,
        p_reff=1.3,
        rand_qtab=False,
        learning_rate=0.5,
        discount_factor=0.6,
        exploration_rate=0.5,
        exploration_decay_rate=0.9,
        k_s=[0.1, 0.5, 1, 2, 7],
        log_level=logging.INFO
    )


def best_combination_experiment(base_folder, env_entry_point, t_s):
    experiment_name = "best_parameters_combination"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - time_step = {}".format(
            i, t) for i, t in enumerate(t_s)])
        f.write("""{}

Changed time step in: 
{}

Other experiment parameters were fixed:
    n_experiments=5,
    n_episodes=200,
    visualize=False,
    max_n_steps=200//time_step, # modelling 400 seconds horizon
    p_reff=1.2,
    rand_qtab=False,
    learning_rate=0.5,
    discount_factor=0.6,
    k_s=[0.1, 0.5, 1, 2, 7],
    exploration_rate=0.5,
    exploration_decay_rate=0.9,   
    log_level=logging.INFO,
    p_reff_amplitude=0,
    p_reff_period=200,
        """.format(experiment_name, param_i_correspondence))

    for t in t_s:
        subfolder = "{}/time_step={}".format(experiment_folder, t)
        os.mkdir(subfolder)
        max_n_steps = 200 // t
        run_experiment_with_result_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_experiments=5,
            n_episodes=100,
            visualize=False,
            max_n_steps=max_n_steps,
            time_step=t,
            p_reff=1.2,
            rand_qtab=False,
            learning_rate=0.5,
            discount_factor=0.6,
            exploration_rate=0.5,
            exploration_decay_rate=0.9,
            k_s=[0.1, 0.5, 1, 2, 7],
            log_level=logging.INFO,
            p_reff_amplitude=0,
            p_reff_period=200
        )


def validate_stochastic_experiment(base_folder, const_actions, env_entry_point):
    experiment_name = "stoch-env_validation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - {}".format(i, k_s) for i, k_s in enumerate(const_actions)])
        f.write("""{}

Changed constant actions in: 
{}

Other experiment parameters were fixed:
        n_episodes=10,
        max_n_steps=20,
        time_step=1,
        p_reff=1.3,
        const_action=a,
        log_level=logging.INFO,
        p_reff_amplitude=0.3,
        p_reff_period=200,
        get_seed=lambda: 23
            """.format(experiment_name, param_i_correspondence))

    for a in const_actions:
        subfolder = "{}/k={}".format(experiment_folder, a)
        os.mkdir(subfolder)
        run_baseline_experiment_with_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_episodes=3,
            max_n_steps=200,
            time_step=1,
            p_reff=1.3,
            const_action=a,
            log_level=logging.INFO,
            p_reff_amplitude=0,
            p_reff_period=200,
            get_seed=lambda: 23
        )


if __name__ == "__main__":
    import time
    start = time.time()
    folder = "stochastic_ps_experiments"
    env_entry_point = "examples:JModelicaCSPSEnv"
    dym_env_class = "examples:DymCSConfigurablePSEnv"
    stoch_env = "examples:JMCSPSStochasticEnv"

    # following experiments rake significant amount of time, so it is advised to run only one of them at once
    # 1
    # ks_experiment(folder, ks=[
    #    [1, 2, 3],
    #    [0.5, 1, 2, 3, 4],
    #    [0.1, 0.5, 1, 2, 7]
    # ], env_entry_point=env_entry_point)

    # 2
    # exploration_experiment(folder, explor_params=[[1, 0.99],
    #                                              [0.2, 1]], env_entry_point=env_entry_point)

    # 3
    # timestep_experiment(folder, [2, 5], env_entry_point=env_entry_point)

    # 4
    # learning_rate_experiment(folder, [0.2, 0.8], env_entry_point=env_entry_point)

    # 5
    # discount_factor_experiment(folder, [0.2, 0.9], env_entry_point=env_entry_point)

    # 6
    # baseline_experiment(folder, [0.5, 1, 2, 5], env_entry_point=stoch_env)

    # run after validation of updated class structure is done:

    # 7
    # reward_experiment(folder, env_entry_point, [
    #    ["-MSEx100", lambda u, p: -100*(u-p)**2],
    #    ["-MAEx100", lambda u, p: -100 * abs(u - p)]
    # ])

    # 8
    best_combination_experiment(folder, env_entry_point=stoch_env, t_s=[1, 5])

    # validate_stochastic_experiment(folder, [0.5, 1, 2, 5], stoch_env)
    # validation_experiment(folder, env_entry_point=env_entry_point)

    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
