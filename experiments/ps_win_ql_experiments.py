import logging
import os

from agents.ps_win_osp_q_learning import ps_train_test_window_osp_ql
from experiment_pipeline import prepare_experiment, run_ps_agent_experiment_with_result_files, suppress_console


def gen_exp_descr(experiment_name, param_i_correspondence):
    descr = """{}
    {}
    
    Other parameters were fixed to default values:
    n_repeat={},
    n_episodes={},
    visualize={},
    max_n_steps={},
    time_step={},
    p_reff={},
    rand_qtab={},
    learning_rate={},
    discount_factor={},
    exploration_rate={},
    exploration_decay_rate={},
    k_s={},
    n_bins={},
    window_size={}    
    log_level=logging.INFO
    """.format(experiment_name, param_i_correspondence, N_REPEAT, N_EPISODES, VISUALIZE, MAX_NUMBER_OF_STEPS,
               TIME_STEP, P_REF, RAND_QTAB, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_DECAY_RATE, EXPLORATION_RATE,
               ACTIONS, N_BINS, WINDOWS_SIZE)

    return descr


def hop_window_experiment(base_folder, env_entry_point, ws_s,
                          experiment_name="hop_window", write_large=True, **kwargs):

    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - window size = {}".format(
            i, ws) for i, ws in enumerate(ws_s)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for ws in ws_s:
        subfolder = "{}/window_size={}".format(experiment_folder, ws)
        os.mkdir(subfolder)

        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_window_osp_ql(
                ps_env=env,
                max_number_of_steps=MAX_NUMBER_OF_STEPS,
                n_episodes=N_EPISODES,
                rand_qtab=RAND_QTAB,
                learning_rate=LEARNING_RATE,
                discount_factor=DISCOUNT_FACTOR,
                exploration_rate=EXPLORATION_RATE,
                exploration_decay_rate=EXPLORATION_DECAY_RATE,
                k_s=ACTIONS,
                visualize=VISUALIZE,
                n_bins=N_BINS,
                window_size=ws,
                n_test_episodes=N_TEST_EPISODES,
                n_test_steps=N_TEST_STEPS,
                hop_size=ws
            ),
            base_folder=subfolder,
            n_repeat=N_REPEAT,
            time_step=TIME_STEP,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=None,
            write_large=write_large,
            **kwargs
        )


def slide_window_experiment(base_folder, env_entry_point, ws_s,
                          experiment_name="slide_window", write_large=True, **kwargs):

    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - window size = {}".format(
            i, ws) for i, ws in enumerate(ws_s)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for ws in ws_s:
        subfolder = "{}/window_size={}".format(experiment_folder, ws)
        os.mkdir(subfolder)

        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_window_osp_ql(
                ps_env=env,
                max_number_of_steps=MAX_NUMBER_OF_STEPS,
                n_episodes=N_EPISODES,
                rand_qtab=RAND_QTAB,
                learning_rate=LEARNING_RATE,
                discount_factor=DISCOUNT_FACTOR,
                exploration_rate=EXPLORATION_RATE,
                exploration_decay_rate=EXPLORATION_DECAY_RATE,
                k_s=ACTIONS,
                visualize=VISUALIZE,
                n_bins=N_BINS,
                window_size=ws,
                n_test_episodes=N_TEST_EPISODES,
                n_test_steps=N_TEST_STEPS
            ),
            base_folder=subfolder,
            n_repeat=N_REPEAT,
            time_step=TIME_STEP,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=None,
            write_large=write_large,
            **kwargs
        )


if __name__ == "__main__":
    import time
    start = time.time()
    stoch_folder = "./results/ps_stoch_adv/"

    stoch_env = "experiment_pipeline:JMCSPSStochasticEnv"

    N_REPEAT = 5
    TIME_STEP = 1
    P_REF = 1.2
    LOG_LEVEL = logging.INFO
    VISUALIZE = False
    RAND_QTAB = False
    MAX_NUMBER_OF_STEPS = 200
    N_EPISODES = 100
    N_TEST_EPISODES = 50
    N_TEST_STEPS = None
    LEARNING_RATE = 0.5
    DISCOUNT_FACTOR = 0.6
    EXPLORATION_RATE = 0.5
    EXPLORATION_DECAY_RATE = 0.9
    ACTIONS = [0.1, 0.5, 1, 2, 7]
    BASELINE_ACTIONS = [0.5, 1, 2, 3, 4, 5, 6, 7]
    N_BINS = 100
    WINDOWS_SIZE = 5
    HOP_SIZE = 1

    slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1, 2],
                            experiment_name="slide_win_exp_100_bins")
    slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4], write_large=False,
                            experiment_name="slide_win_exp_100_bins")

    hop_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1, 2],
                            experiment_name="hop_win_exp_100_bins")
    hop_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4], write_large=False,
                            experiment_name="hop_win_exp_100_bins", )

    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
