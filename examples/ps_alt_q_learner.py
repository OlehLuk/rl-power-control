import logging
import gym
import numpy as np
import time

from gymalgs.rl.alt_q_learning import AltQLearner


def mse(a, b):
    return np.mean(np.power(np.array(a)-np.array(b), 2))


def ps_train_qlearning(ps_env, max_number_of_steps=20, n_episodes=4, visualize=True):
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
    n_actions = 8

    episode_lengths = np.array([])
    episodes_mse_reward = np.array([])
    episodes_us = []
    episodes_ps = []

    u_bins = _get_bins(0, 1.7, 100)
    # ref_bins = _get_bins(1.2, 1.4, 1)
    k_s = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]

    learner = AltQLearner(learning_rate=0.5,
                             random_action_rate=0.5,
                             value=0,
                             num_states=100**n_outputs,
                             num_actions=n_actions,
                             discount_factor=0.9)

    for episode in range(n_episodes):
        u, _ = ps_env.reset()
        us = []
        ps = []
        state = _get_state_index([_to_bin(u, u_bins)])

        action = learner.interact(None, state, None) + 1
        for step in range(max_number_of_steps):
            if visualize:
                ps_env.render()
            observation, reward, done, _ = ps_env.step(k_s[action])

            u, p = observation

            us.append(u)
            ps.append(p)

            state_prime = _get_state_index([_to_bin(u, u_bins)])

            action = learner.interact(reward, state_prime, done) + 1
            if done or step == max_number_of_steps - 1:
                episode_lengths = np.append(episode_lengths, int(step + 1))

                episodes_us.append(us)
                episodes_ps.append(ps)
                episodes_mse_reward = np.append(episodes_mse_reward, mse(us, ps))
                break
    end = time.time()
    execution_time = end - start
    return learner, episode_lengths, execution_time, episodes_mse_reward, episodes_us, episodes_ps


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


def run_experiments(n_experiments=1,
                    n_episodes=10,
                    max_n_steps=10,
                    visualize=False,
                    p_reff=1.3,
                    time_step=1,
                    log_level=logging.DEBUG):
    """
    Wrapper for running experiment of q-learning training on PS environment.
    Is responsible for environment creation and closing, sets all necessary parameters of environment.
    Runs n exepriments, where each experiment is training Q-learning agent on the same environment.
    After one agent finished training, environment is reset to the initial state.
    Parameters of the experiment:
    :param p_reff: value of p_reff parameter of the environments
    :param n_episodes: number of episodes to perform in each experiment run
    :param visualize: boolean flag if experiments should be rendered
    :param n_experiments: number of experiments to perform.
    :param log_level: level of logging that should be used by environment during experiments.

    :return: trained Q-learning agent, array of actual episodes length
    that were returned from cart_pole_train_qlearning()
    """
    config = {
        'time_step': time_step,
        'p_reff': p_reff,
        'log_level': log_level
    }

    from gym.envs.registration import register
    env_name = "JModelicaCSPSEnv-v0"

    register(
        id=env_name,
        entry_point='examples:JModelicaCSPSPSEnv',
        kwargs=config
    )
    trained_agent_s =[]
    episodes_length_s = []
    exec_time_s = []
    mse_rewards_s = []
    eps_us = []
    eps_ps = []
    env = gym.make(env_name)
    for i in range(n_experiments):
        trained_agent, episodes_length, exec_time, episodes_mse_reward, ep_us, ep_ps = ps_train_qlearning(
            env,
            n_episodes=n_episodes,
            visualize=visualize,
            max_number_of_steps=max_n_steps)
        trained_agent_s.append(trained_agent)
        episodes_length_s.append(episodes_length)
        exec_time_s.append(exec_time)
        mse_rewards_s.append(episodes_mse_reward)
        eps_us.append(ep_us)
        eps_ps.append(ep_ps)
        env.reset()

    env.close()
    # delete registered environment to avoid errors in future runs.
    del gym.envs.registry.env_specs[env_name]
    return trained_agent_s, episodes_length_s, exec_time_s, mse_rewards_s, eps_us, eps_ps


if __name__ == "__main__":
    n_episodes = 30
    n_steps = 100
    _, episodes_lengths, exec_times, mse_rewards, eps_ps, eps_us = run_experiments(visualize=False,
                                                                                   log_level=logging.INFO,
                                                                                   n_episodes=n_episodes,
                                                                                   max_n_steps=n_steps,
                                                                                   time_step=2)
    print("Experiment length {} s".format(exec_times[0]))
    print(u"Avg episode length {} {} {}".format(episodes_lengths[0].mean(),
                                                chr(177),
                                                episodes_lengths[0].std()))
    print(u"All final reward {}".format(mse_rewards))
    import matplotlib.pyplot as plt
    plt.plot(np.arange(1, n_episodes + 1, 1), mse_rewards[0])
    plt.show()
    for i in [0, 5, 7]:
        plt.plot(np.arange(1, n_steps + 1, 1), eps_ps[0][i])
        plt.plot(np.arange(1, n_steps + 1, 1), eps_us[0][i])
        plt.title("{}-th episode of training".format(i))
        plt.show()



