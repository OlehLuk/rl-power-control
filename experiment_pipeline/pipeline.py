import gym
import logging
import numpy as np
import os
from datetime import datetime
import pickle as pkl

from experiment_pipeline import mse


def run_ps_experiment(experiment_procedure,
                       env_entry_point,
                       compute_reward=None,
                       time_step=1,
                       p_reff=1.3,
                       log_level=logging.INFO,
                       **kwargs):

    """
    Wrapper for running experiment of q-learning training on PS environment.
    Is responsible for environment creation and closing, sets all necessary parameters of environment.
    Runs n exepriments, where each experiment is training Q-learning agent on the same environment.
    After one agent finished training, environment is reset to the initial state.
    Parameters of the experiment:
    :param p_reff: value of p_reff parameter of the environments
    :param n_episodes: number of episodes to perform in each experiment run
    :param n_experiments: number of experiments to perform.
    :param log_level: level of logging that should be used by environment during experiments.

    :return: trained Q-learning agent, array of actual episodes length
    that were returned from cart_pole_train_qlearning()
    """
    config = {
        'time_step': time_step,
        'p_reff': p_reff,
        'log_level': log_level,
        'compute_reward': compute_reward
    }

    for key, value in kwargs.items():
        config.update({key: value})

    from gym.envs.registration import register
    env_name = "CSPSEnv-v0"

    register(
        id=env_name,
        entry_point=env_entry_point,
        kwargs=config
    )
    env = gym.make(env_name)

    result = experiment_procedure(env)
    # delete registered environment to avoid errors in future runs.
    del gym.envs.registry.env_specs[env_name]
    return result


def ps_constant_control_experiment(env, n_episodes, max_number_of_steps, const_action):
    episode_lengths = np.array([])
    episodes_mse_reward = np.array([])
    episodes_us = []
    episodes_ps = []

    for _ in range(n_episodes):
        u, p = env.reset()
        us = [u]
        ps = [p]

        for step in range(max_number_of_steps):
            observation, reward, done, _ = env.step(const_action)
            u, p = observation
            us.append(u)
            ps.append(p)

            if done or step == max_number_of_steps - 1:
                episode_lengths = np.append(episode_lengths, int(step + 1))
                episodes_us.append(us)
                episodes_ps.append(ps)
                episodes_mse_reward = np.append(episodes_mse_reward, mse(us, ps))
                break

    return episode_lengths, episodes_mse_reward, episodes_us, episodes_ps


def run_ps_const_control_experiment(env_entry_point,
                                     n_episodes=10,
                                     max_n_steps=20,
                                     const_action=1,
                                     log_level=logging.INFO,
                                     p_reff=1.3,
                                     time_step=1,
                                     **kwargs):
    return run_ps_experiment(lambda env: ps_constant_control_experiment(
        env=env,
        n_episodes=n_episodes,
        max_number_of_steps=max_n_steps,
        const_action=const_action

    ),
        env_entry_point=env_entry_point,
        time_step=time_step,
        p_reff=p_reff,
        log_level=log_level,
        **kwargs)


def run_ps_const_control_experiment_with_files(base_folder,
                                               const_action,
                                               n_episodes,
                                               max_n_steps,
                                               p_reff,
                                               time_step,
                                               log_level,
                                               env_entry_point,
                                               **kwargs):
    episode_lengths, episodes_mse_reward, episodes_us, episodes_ps = run_ps_const_control_experiment(
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


def agent_experiment(env, agent_args, agent_train_test_once, n_repeat=1):
    trained_agent_s = []
    episodes_length_s = []
    exec_time_s = []
    mse_rewards_s = []
    eps_us = []
    eps_ps = []
    eps_acs = []
    expl_perfs = []
    expl_acts = []

    for _ in range(n_repeat):
        trained_agent, episodes_length, exec_time, episodes_mse_reward, ep_us, ep_ps, ep_ac, expl_perf, expl_act = \
            agent_train_test_once(
                env,
                **agent_args
            )
        trained_agent_s.append(trained_agent)
        episodes_length_s.append(episodes_length)
        exec_time_s.append(exec_time)
        mse_rewards_s.append(episodes_mse_reward)
        eps_us.append(ep_us)
        eps_ps.append(ep_ps)
        eps_acs.append(ep_ac)
        expl_perfs.append(expl_perf)
        expl_acts.append(expl_act)
        env.reset()

    return trained_agent_s, episodes_length_s, exec_time_s, mse_rewards_s, eps_us, eps_ps, eps_acs, expl_perfs, expl_acts


def run_ps_agent_experiment(env_entry_point,
                            agent_train_test_once,
                            agent_args,
                            n_repeat=1,
                            p_reff=1.3,
                            time_step=1,
                            log_level=logging.INFO,
                            compute_reward=None,
                            **kwargs):
    return run_ps_experiment(lambda env: agent_experiment(
        env=env,
        agent_args=agent_args,
        agent_train_test_once=agent_train_test_once,
        n_repeat=n_repeat
    ),
        compute_reward=compute_reward,
        env_entry_point=env_entry_point,
        time_step=time_step,
        p_reff=p_reff,
        log_level=log_level,
        **kwargs)


def run_ps_agent_experiment_with_result_files(base_folder,
                                              n_repeat,
                                              agent_train_test_once,
                                              agent_args,
                                              time_step,
                                              p_reff,
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
        run_ps_agent_experiment(
            env_entry_point=env_entry_point,
            n_experiments=n_repeat,
            agent_train_test_once=agent_train_test_once,
            agent_args=agent_args,
            p_reff=p_reff,
            time_step=time_step,
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

    for i in range(n_repeat):
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


def prepare_experiment(base_folder, experiment_name):
    now = datetime.now()
    experiment_folder = "{}/{}_{}".format(base_folder, experiment_name, now.strftime("%d-%m-%Y_%H:%M"))

    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    else:
        print("Terminating: possible override of previous experiment results is detected.")
        return None

    return experiment_folder


def baseline_experiment(base_folder, const_actions, env_entry_point,
                        experiment_name="constant_action_variation"):

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
        run_ps_const_control_experiment_with_files(
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
        run_ps_agent_experiment_with_result_files(
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


