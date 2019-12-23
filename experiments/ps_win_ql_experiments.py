import logging
import os

from agents.ps_win_osp_q_learning import ps_train_test_window_osp_ql
from experiment_pipeline import prepare_experiment, run_ps_agent_experiment_with_result_files, suppress_console
from experiments.discr_bins_calculation import load_bins


def gen_exp_descr(experiment_name, param_i_correspondence, **kwargs):
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
    window_size={},
    bins={},   
    log_level=logging.INFO
    {}
    """.format(experiment_name, param_i_correspondence, N_REPEAT, N_EPISODES, VISUALIZE, MAX_NUMBER_OF_STEPS,
               TIME_STEP, P_REF, RAND_QTAB, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_DECAY_RATE, EXPLORATION_RATE,
               ACTIONS, N_BINS, WINDOWS_SIZE, BINS, kwargs)

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
                hop_size=ws,
                bins=BINS
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
        f.write(gen_exp_descr(experiment_name, param_i_correspondence, **kwargs))

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
                bins=BINS
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
    BINS = None
    N_BINS = 10

    with suppress_console():
        N_EPISODES = 200
        BINS = load_bins("results/dicretization_bins/baseline_p_ref_10bins.csv")
        slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
                               experiment_name="ql_baseline_p_ref_10bins_200ep")




        # N_EPISODES = 200
        # BINS = load_bins("results/dicretization_bins/baseline12_hist_bins.csv")
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
        #                        experiment_name="ql_baseline12_hist_bins")

        # BINS = load_bins("results/dicretization_bins/bas_skip175_hist_bins.csv")
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
        #                        experiment_name="ql_baseline12_skip175_hist_bins_skip175",
        #                        simulation_start_time=SKIP_SECONDS)
        # BINS = load_bins("results/dicretization_bins/bas_skip175_p_ref_10bins.csv")
        # hop_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[2, 4],
        #                       experiment_name="hop_win_skip175_p_ref_10bins",
        #                       simulation_start_time=SKIP_SECONDS)
        # BINS = load_bins("results/dicretization_bins/bas_skip175_p_ref_30bins.csv")
        # hop_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[2, 4],
        #                       experiment_name="hop_win_skip175_p_ref_30bins",
        #                       simulation_start_time=SKIP_SECONDS)
        # BINS = load_bins("results/dicretization_bins/bas_skip175_p_ref_30bins.csv")
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[2, 4],
        #                         experiment_name="slide_win_skip175_p_ref_30bins",
        #                         simulation_start_time=SKIP_SECONDS)

        # BINS = None
        # N_BINS = 25
        # N_TEST_STEPS = 400
        # N_EPISODES = 200
        # SKIP_SECONDS = 175
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[2],
        #                        experiment_name="slide_skip175_200ep_long_test_25bins",
        #                        simulation_start_time=SKIP_SECONDS)



        # SKIP_SECONDS = 175
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[2],
        #                        experiment_name="slide_skip175_long_test_25bins_09-17",
        #                        simulation_start_time=SKIP_SECONDS)

        # BINS = load_bins("results/dicretization_bins/ql12_32ep_p_ref_10bins.csv")
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
        #                        experiment_name="ql_ql12_32ep_p_ref_10bins_skip175",
        #                        simulation_start_time=SKIP_SECONDS)

        # BINS = load_bins("results/dicretization_bins/baseline_p_ref_10bins.csv")
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
        #                       experiment_name="ql_baseline_p_ref_10bins_skip175",
        #                        simulation_start_time=SKIP_SECONDS)

        # SKIP_SECONDS = 175
        # BINS = load_bins("results/dicretization_bins/baseline12_quantile_10bins.csv")
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
        #                        experiment_name="ql_baseline12_quantile_10bins_skip_175",
        #                        simulation_start_time=SKIP_SECONDS)

        # BINS = load_bins("results/dicretization_bins/baseline12_hist_bins.csv")
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
        #                        experiment_name="ql_baseline12_hist_bins_skip_175",
        #                        simulation_start_time=SKIP_SECONDS)


        # BINS = load_bins("results/dicretization_bins/baseline12_quantile_25bins.csv")
        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
        #                        experiment_name="baseline_quantile_bins_test")


        # slide_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4], write_large=False,
        #                        experiment_name="slide_win_exp_100_09-17_bins", simulation_start_time=SKIP_SECONDS)

        # hop_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[2],
        #                        experiment_name="hop_win_exp_100_09-17_bins_long")
        # hop_window_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4], write_large=False,
        #                        experiment_name="hop_win_exp_10_bins")

    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
