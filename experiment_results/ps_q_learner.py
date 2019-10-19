import logging
import gym
from modelicagym.gymalgs.rl import QLearner
import numpy as np
import time


def mse(a, b):
    return np.mean(np.power(np.array(a)-np.array(b), 2))


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


def constant_agent_experiment(env, n_episodes, max_number_of_steps, const_action):
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


def run_ps_experiments(experiment_procedure,
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
        # entry_point='experiment_results:JModelicaCSPSEnv',
        # entry_point='experiment_results:DymCSConfigurablePSEnv',
        kwargs=config
    )
    env = gym.make(env_name)

    result = experiment_procedure(env)
    # delete registered environment to avoid errors in future runs.
    del gym.envs.registry.env_specs[env_name]
    return result


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


def run_ps_const_agent_experiments(env_entry_point,
                                    n_episodes=10,
                                    max_n_steps=20,
                                    const_action=1,
                                    log_level=logging.INFO,
                                    p_reff=1.3,
                                    time_step=1,
                                    **kwargs):
    return run_ps_experiments(lambda env: constant_agent_experiment(
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


if __name__ == "__main__":
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

    import matplotlib.pyplot as plt
    plt.plot(np.arange(1, n_episodes + 1, 1), mse_rewards[0])
    plt.show()
    for i in [0, 5, 9]:
        plt.plot(np.arange(0, n_steps + 1, 1), eps_ps[0][i])
        plt.plot(np.arange(0, n_steps + 1, 1), eps_us[0][i])
        plt.title("{}-th episode of training".format(i))
        plt.show()



