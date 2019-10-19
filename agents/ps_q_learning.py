import time
import numpy as np
from modelicagym.gymalgs.rl import QLearner
from experiment_pipeline import mse


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
    n_outputs = 1
    n_actions = len(k_s)

    u_bins = _get_bins(0.1, 1.7, 100)

    # ref_bins = _get_bins(1.2, 1.4, 1)

    learner = QLearner(n_states=100**n_outputs,
                       n_actions=n_actions,
                       learning_rate=learning_rate,
                       discount_factor=discount_factor,
                       exploration_rate=exploration_rate,
                       exploration_decay_rate=exploration_decay_rate,
                       rand_qtab=rand_qtab)

    trained_agent, episode_lengths, episodes_mse_reward, episodes_us, episodes_ps, episodes_actions = \
        go_n_episodes_with_agent(ps_env, learner, n_episodes,
                                 max_number_of_steps, u_bins,
                                 visualize, k_s, test_performance=False)

    _, _, exploit_performance, _, _, exploit_actions = \
        go_n_episodes_with_agent(ps_env, trained_agent, n_episodes,
                                 max_number_of_steps, u_bins,
                                 visualize, k_s, test_performance=True)

    end = time.time()
    execution_time = end - start
    return trained_agent, episode_lengths, execution_time, episodes_mse_reward,\
           episodes_us, episodes_ps, episodes_actions, exploit_performance, exploit_actions


# Internal logic for state discretization
def _get_bins(lower_bound, upper_bound, n_bins):
    """
    Given bounds for environment state variable splits it into n_bins number of bins,
    taking into account possible values outside the bounds.

    :param lower_bound: lower bound for variable describing state space
    :param upper_bound: upper bound for variable describing state space
    :param n_bins: number of bins to receive
    :return: n_bins-1 values separating bins. I.e. the most left bin is opened from the left,
    the most right bin is open from the right.
    """
    return np.linspace(lower_bound, upper_bound, n_bins + 1)[1:-1]


def _to_bin(value, bins):
    """
    Transforms actual state variable value into discretized one,
    by choosing the bin in variable space, where it belongs to.

    :param value: variable value
    :param bins: sequence of values separating variable space
    :return: number of bin variable belongs to. If it is smaller than lower_bound - 0.
    If it is bigger than the upper bound
    """
    return np.digitize(x=[value], bins=bins)[0]


def _get_state_index(state_bins):
    """
    Transforms discretized environment state (represented as sequence of bin indexes) into an integer value.
    Value is composed by concatenating string representations of a state_bins.
    Received string is a valid integer, so it is converted to int.

    :param state_bins: sequence of integers that represents discretized environment state.
    Each integer is an index of bin, where corresponding variable belongs.
    :return: integer value corresponding to the environment state
    """
    state = int("".join(map(lambda state_bin: str(state_bin), state_bins)))
    return state


def go_n_episodes_with_agent(ps_env, agent, n_episodes,
                             max_number_of_steps, u_bins,
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

        state_idx = _get_state_index([_to_bin(u, u_bins)])

        action_idx = agent.set_initial_state(state_idx)
        episode_action = [actions[action_idx]]
        for step in range(max_number_of_steps):
            if visualize:
                ps_env.render()
            observation, reward, done, _ = ps_env.step(actions[action_idx])

            u, p = observation

            us.append(u)
            ps.append(p)

            state_prime = _get_state_index([_to_bin(u, u_bins)])

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