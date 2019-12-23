import time
import numpy as np
from modelicagym.gymalgs.rl import QLearner
from .discretization import get_state_index, to_bin, get_bins
from experiment_pipeline import mse, ProgressBar


# osp - one state parameter
def ps_train_test_window_osp_ql(ps_env,
                                max_number_of_steps=20,
                                n_episodes=4,
                                rand_qtab=False,
                                learning_rate=0.5,
                                discount_factor=0.6,
                                exploration_rate=0.5,
                                exploration_decay_rate=0.99,
                                k_s=(1, 2, 3),
                                visualize=True,
                                window_size=1,
                                hop_size=1,
                                n_bins=100,
                                n_test_episodes=None,
                                n_test_steps=None,
                                bins=None):
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
    n_outputs = window_size
    n_actions = len(k_s)

    if bins is None:
        u_bins = get_bins(0.9, 1.7, n_bins)
    else:
        u_bins = bins
        n_bins = len(u_bins) + 1
    # ref_bins = _get_bins(1.2, 1.4, 1)

    learner = QLearner(n_states=n_bins**n_outputs,
                       n_actions=n_actions,
                       learning_rate=learning_rate,
                       discount_factor=discount_factor,
                       exploration_rate=exploration_rate,
                       exploration_decay_rate=exploration_decay_rate,
                       rand_qtab=rand_qtab)

    trained_agent, episode_lengths, episodes_mse_reward, episodes_us, episodes_ps, episodes_actions = \
        go_n_episodes_with_agent(ps_env, learner, n_episodes,
                                 max_number_of_steps, u_bins,
                                 visualize, k_s, test_performance=False,
                                 window_size=window_size, hop_size=hop_size)

    if n_test_episodes is None:
        n_test_episodes = n_episodes

    if n_test_steps is None:
        n_test_steps = max_number_of_steps

    _, _, exploit_performance, _, _, exploit_actions = \
        go_n_episodes_with_agent(ps_env, trained_agent, n_test_episodes,
                                 n_test_steps, u_bins,
                                 visualize, k_s, test_performance=True,
                                 window_size=window_size, hop_size=hop_size)

    end = time.time()
    execution_time = end - start
    return trained_agent, episode_lengths, execution_time, episodes_mse_reward,\
           episodes_us, episodes_ps, episodes_actions, exploit_performance, exploit_actions


def go_n_episodes_with_agent(ps_env, agent, n_episodes,
                             max_number_of_steps, u_bins,
                             visualize, actions, window_size=1, hop_size=1,
                             test_performance=False):

    episode_lengths = np.array([])
    episodes_mse_reward = np.array([])
    episodes_us = []
    episodes_ps = []
    episodes_actions = []

    pb_name = "n_episode in train"

    if test_performance:
        agents_rand_act_rate = agent.random_action_rate
        agent.random_action_rate = 0
        pb_name = "n_episode in test"

    pb = ProgressBar(pb_name, n_episodes)

    n_bins = len(u_bins) + 1

    for _ in range(n_episodes):
        us = []
        ps = []
        window_us = []
        u, p = ps_env.reset()
        us.append(u)
        ps.append(p)
        window_us.append(to_bin(u, u_bins))
        episode_action = []

        for _ in range(1, window_size):
            episode_action.append(0)
            observation, reward, done, _ = ps_env.step(0)
            u, p = observation
            us.append(u)
            ps.append(p)
            window_us.append(to_bin(u, u_bins))

        state_idx = get_state_index(window_us, n_bins)

        action_idx = agent.set_initial_state(state_idx)

        hop_flag = hop_size

        for step in range(window_size, max_number_of_steps+1):
            if visualize:
                ps_env.render()

            episode_action.append(actions[action_idx])
            observation, reward, done, _ = ps_env.step(actions[action_idx])
            u, p = observation
            us.append(u)
            ps.append(p)
            window_us.append(to_bin(u, u_bins))
            window_us.pop(0)

            hop_flag -= 1

            if hop_flag == 0:
                hop_flag = hop_size
                state_prime = get_state_index(window_us, n_bins)

                if test_performance:
                    action_idx = agent.use(state_prime)
                else:
                    action_idx = agent.learn_observation(state_prime, reward)

            if done or step == max_number_of_steps:
                episode_lengths = np.append(episode_lengths, int(step))

                episodes_us.append(us)
                episodes_ps.append(ps)
                episodes_actions.append(episode_action)
                episodes_mse_reward = np.append(episodes_mse_reward, mse(us, ps))
                pb.step()
                break

    if test_performance:
        agent.random_action_rate = agents_rand_act_rate
        # print(episodes_actions)

    pb.close()

    return agent, episode_lengths, episodes_mse_reward, episodes_us, episodes_ps, episodes_actions
