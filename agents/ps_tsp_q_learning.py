import time
import numpy as np
from modelicagym.gymalgs.rl import QLearner

from .discretization import get_state_index, to_bin, get_bins
from experiment_pipeline import mse


# tsp - two state parameters
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

        state_idx = get_state_index([to_bin(p, p_bins),
                                     to_bin(u, u_bins)])

        action_idx = agent.set_initial_state(state_idx)
        episode_action = [actions[action_idx]]
        for step in range(max_number_of_steps):
            if visualize:
                ps_env.render()
            observation, reward, done, _ = ps_env.step(actions[action_idx])

            u, p = observation

            us.append(u)
            ps.append(p)

            state_prime = get_state_index([to_bin(p, p_bins),
                                           to_bin(u, u_bins)])

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


def ps_train_test_tsp_ql(ps_env,
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

    u_bins = get_bins(0.1, 1.7, 100)
    p_bins = get_bins(1.2, 1.4, 2)

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