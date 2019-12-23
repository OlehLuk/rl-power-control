import random
import time
import numpy as np
from .dqn import DqnAgent
from experiment_pipeline import mse, ProgressBar


def ps_train_test_dqn(ps_env,
                      max_number_of_steps=20,
                      n_episodes=5,

                      learning_rate=0.01,
                      discount_factor=0.6,
                      exploration_rate=0.5,
                      exploration_decay_rate=0.99,
                      expl_rate_final=0.05,
                      k_s=(0.1, 0.5, 1, 2, 7),
                      window_size=4,
                      n_test_episodes=None,
                      n_test_steps=None,
                      use_all_rewards=False,
                      buffer_size=8, batch_size=8):
    """
    Runs one experiment of DQN training on power system environment
    :param ps_env: environment RL agent will learn on.
    :param max_number_of_steps: maximum episode length.
    :param n_episodes: number of episodes to perform.
    :param visualize: flag if experiments should be rendered.
    :return: trained Q-learning agent, array of actual episodes length,
    execution time in s, mse-rewards at the end of each episode
    """

    start = time.time()

    agent = DqnAgent(actions=k_s,
                     n_hidden_1=32,
                     n_hidden_2=32,
                     n_state_variables=window_size,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     exploration_rate=exploration_rate,
                     expl_rate_decay=exploration_decay_rate,
                     discount_factor=discount_factor,
                     expl_rate_final=expl_rate_final
                     )

    trained_agent, episode_lengths, episodes_mse_reward, episodes_us, episodes_ps, episodes_actions = \
        go_n_episodes_with_agent(ps_env, agent, n_episodes,
                                 max_number_of_steps, k_s, test_performance=False,
                                 window_size=window_size)

    if n_test_episodes is None:
        n_test_episodes = n_episodes

    if n_test_steps is None:
        n_test_steps = max_number_of_steps

    _, _, exploit_performance, _, _, exploit_actions = \
        go_n_episodes_with_agent(ps_env, trained_agent, n_test_episodes,
                                 n_test_steps, k_s, test_performance=True,
                                 window_size=window_size)

    end = time.time()
    execution_time = end - start
    return trained_agent, episode_lengths, execution_time, episodes_mse_reward,\
           episodes_us, episodes_ps, episodes_actions, exploit_performance, exploit_actions


def go_n_episodes_with_agent(ps_env, agent, n_episodes,
                             max_number_of_steps,
                             actions, window_size=1,
                             test_performance=False):
    hop_size = window_size
    episode_lengths = np.array([])
    episodes_mse_reward = np.array([])
    episodes_us = []
    episodes_ps = []
    episodes_actions = []

    pb_name = "n_episode in train"

    if test_performance:
        pb_name = "n_episode in test"

    pb = ProgressBar(pb_name, n_episodes)

    for _ in range(n_episodes):
        us = []
        ps = []
        window_us = []
        u, p = ps_env.reset()
        us.append(u)
        ps.append(p)
        window_us.append(u)
        action = random.sample(actions)
        episode_action = [action]
        rewards = [0]

        for _ in range(1, window_size):
            episode_action.append(action)
            observation, reward, done, _ = ps_env.step(action)
            u, p = observation
            us.append(u)
            ps.append(p)
            window_us.append(u)
            rewards.append(reward)

        state = window_us
        hop_flag = hop_size

        for step in range(window_size, max_number_of_steps+1):
            episode_action.append(action)
            observation, reward, done, _ = ps_env.step(action)
            u, p = observation
            us.append(u)
            ps.append(p)
            window_us.append(u)
            window_us.pop(0)
            rewards.append(reward)
            rewards.pop(0)

            hop_flag -= 1

            if hop_flag == 0:
                hop_flag = hop_size
                next_state = window_us
                reward = sum(rewards) / window_size

                if test_performance:
                    action = agent.use(next_state)
                else:
                    action = agent.learn(state, reward, next_state, done)
                    state = next_state

            if done or step == max_number_of_steps:
                episode_lengths = np.append(episode_lengths, int(step))

                episodes_us.append(us)
                episodes_ps.append(ps)
                episodes_actions.append(episode_action)
                episodes_mse_reward = np.append(episodes_mse_reward, mse(us, ps))
                pb.step()
                break

    pb.close()

    return agent, episode_lengths, episodes_mse_reward, episodes_us, episodes_ps, episodes_actions
