import logging
import os
from experiment_results import prepare_experiment
import numpy as np
from experiment_results.ps_q_learner import run_ps_experiments, _get_state_index, _get_bins, _to_bin
import pickle as pkl

from modelicagym.gymalgs.rl import QLearner


def mse(a, b):
    return np.mean(np.power(np.array(a)-np.array(b), 2))


def go_n_episodes_with_agent(ps_env, agent, n_episodes,
                             max_number_of_steps, u_bins, p_bins,
                             visualize, actions, test_performance=False):

    episode_lengths = np.array([])
    episodes_mse_reward = np.array([])
    episodes_us = []
    episodes_ps = []
    episodes_actions = []

    if test_performance:
        agents_rand_act_rate = agent.random_action_rate
        agent.random_action_rate = 0

    for _ in range(n_episodes):
        u, p = ps_env.reset()
        us = [u]
        ps = [p]

        state_idx = _get_state_index([_to_bin(p, p_bins),
                                      _to_bin(u, u_bins)])

        action_idx = agent.set_initial_state(state_idx)
        episode_action = [actions[action_idx]]
        for step in range(max_number_of_steps):
            if visualize:
                ps_env.render()
            observation, reward, done, _ = ps_env.step(actions[action_idx])

            u, p = observation

            us.append(u)
            ps.append(p)

            state_prime = _get_state_index([_to_bin(p, p_bins),
                                            _to_bin(u, u_bins)])

            if test_performance:
                action_idx = agent.use(state_prime)
            else:
                action_idx = agent.learn_observation(state_prime, reward)

            episode_action.append(actions[action_idx])
            if done or step == max_number_of_steps - 1:
                episode_lengths = np.append(episode_lengths, int(step + 1))

                episodes_us.append(us)
                episodes_ps.append(ps)
                episodes_actions.append(episode_action)
                episodes_mse_reward = np.append(episodes_mse_reward, mse(us, ps))
                break

    if test_performance:
        agent.random_action_rate = agents_rand_act_rate
        # print(episodes_actions)

    return agent, episode_lengths, episodes_mse_reward, episodes_us, episodes_ps, episodes_actions


def ps_train_qlearning(ps_env,
                       max_number_of_steps=20,
                       n_episodes=4,
                       rand_qtab=False,
                       learning_rate=0.5,
                       discount_factor=0.6,
                       exploration_rate=0.5,
                       exploration_decay_rate=0.99,
                       k_s=(1, 2, 3),
                       visualize=True):
    """
    Runs one experiment of Q-learning training on power system environment
    :param ps_env: environment RL agent will learn on.
    :param max_number_of_steps: maximum episode length.
    :param n_episodes: number of episodes to perform.
    :param visualize: flag if experiments should be rendered.
    :return: trained Q-learning agent, array of actual episodes length,
    execution time in s, mse-rewards at the end of each episode
    """

    start = time.time()
    n_outputs = 2
    n_actions = len(k_s)

    u_bins = _get_bins(0.1, 1.7, 100)
    p_bins = _get_bins(1.2, 1.4, 2)

    learner = QLearner(n_states=100*2,
                       n_actions=n_actions,
                       learning_rate=learning_rate,
                       discount_factor=discount_factor,
                       exploration_rate=exploration_rate,
                       exploration_decay_rate=exploration_decay_rate,
                       rand_qtab=rand_qtab)

    trained_agent, episode_lengths, episodes_mse_reward, episodes_us, episodes_ps, episodes_actions = \
        go_n_episodes_with_agent(ps_env, learner, n_episodes,
                                 max_number_of_steps, u_bins, p_bins,
                                 visualize, k_s, test_performance=False)

    _, _, exploit_performance, _, _, exploit_actions = \
        go_n_episodes_with_agent(ps_env, trained_agent, n_episodes,
                                 max_number_of_steps, u_bins, p_bins,
                                 visualize, k_s, test_performance=True)

    end = time.time()
    execution_time = end - start
    return trained_agent, episode_lengths, execution_time, episodes_mse_reward,\
           episodes_us, episodes_ps, episodes_actions, exploit_performance, exploit_actions


def ql_experiment(env,
                    n_experiments=1,
                    n_episodes=10,
                    max_n_steps=10,
                    visualize=False,
                    rand_qtab=False,
                    learning_rate=0.5,
                    discount_factor=0.6,
                    exploration_rate=0.5,
                    exploration_decay_rate=0.99,
                    k_s=(1, 2, 3)
                  ):

    trained_agent_s = []
    episodes_length_s = []
    exec_time_s = []
    mse_rewards_s = []
    eps_us = []
    eps_ps = []
    eps_acs = []
    expl_perfs = []
    expl_acts = []

    for _ in range(n_experiments):
        trained_agent, episodes_length, exec_time, episodes_mse_reward, ep_us, ep_ps, ep_ac, expl_perf, expl_act = \
            ps_train_qlearning(
                env,
                n_episodes=n_episodes,
                visualize=visualize,
                max_number_of_steps=max_n_steps,
                rand_qtab=rand_qtab,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                exploration_rate=exploration_rate,
                exploration_decay_rate=exploration_decay_rate,
                k_s=k_s,
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


def run_ps_ql_experiments(env_entry_point,
                          n_experiments=1,
                          n_episodes=10,
                          visualize=False,
                          max_n_steps=20,
                          p_reff=1.3,
                          time_step=1,
                          rand_qtab=False,
                          learning_rate=0.5,
                          discount_factor=0.6,
                          exploration_rate=0.5,
                          exploration_decay_rate=0.99,
                          k_s=(1, 2, 3),
                          log_level=logging.INFO,
                          compute_reward=None,
                          **kwargs):
    return run_ps_experiments(lambda env: ql_experiment(
        env=env,
        n_experiments=n_experiments,
        n_episodes=n_episodes,
        max_n_steps=max_n_steps,
        visualize=visualize,
        rand_qtab=rand_qtab,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        exploration_decay_rate=exploration_decay_rate,
        k_s=k_s,

    ),
        compute_reward=compute_reward,
        env_entry_point=env_entry_point,
        time_step=time_step,
        p_reff=p_reff,
        log_level=log_level,
        **kwargs)


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


def best_combination_experiment(base_folder, env_entry_point, t_s):
    experiment_name = "step_once_consid_p_reff"
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
    p_reff=1.3,
    rand_qtab=False,
    learning_rate=0.5,
    discount_factor=0.6,
    k_s=[0.1, 0.5, 1, 2, 7],
    exploration_rate=0.5,
    exploration_decay_rate=0.9,   
    log_level=logging.INFO,
    p_reff_amplitude=0.2,
    p_reff_period=40
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
            p_reff=1.1,
            rand_qtab=False,
            learning_rate=0.5,
            discount_factor=0.6,
            exploration_rate=0.5,
            exploration_decay_rate=0.9,
            k_s=[0.1, 0.5, 1, 2, 7],
            log_level=logging.INFO,
            p_reff_amplitude=0.3,
            p_reff_period=201,
            # path="../resources/jmodelica/linux/PS_ampl_det.fmu"
        )


if __name__ == "__main__":
    import time
    start = time.time()
    folder = "ampl_stoch_ps_exp_results"
    env_entry_point = "experiment_pipeline:JModelicaCSPSEnv"
    # dym_env_class = "experiment_pipeline:DymCSConfigurablePSEnv"
    stoch_env = "experiment_pipeline:JMCSPSStochasticEnv"
    best_combination_experiment(folder, env_entry_point=stoch_env, t_s=[1, 5])

    # baseline_experiment(folder, [0.5, 1, 2, 5], env_entry_point)
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
