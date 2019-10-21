import logging
import os

from agents import ps_train_test_tsp_ql
from experiment_pipeline import run_ps_agent_experiment_with_result_files, prepare_experiment, baseline_experiment


def best_combination_experiment_tsp(base_folder, env_entry_point, t_s,
                                    experiment_name = "step_once_consid_p_reff", **kwargs):

    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - time_step = {}".format(
            i, t) for i, t in enumerate(t_s)])
        from experiments.ps_ql_experiments import gen_exp_descr
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for t in t_s:
        subfolder = "{}/time_step={}".format(experiment_folder, t)
        os.mkdir(subfolder)
        max_n_steps = MAX_NUMBER_OF_STEPS // t
        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_tsp_ql(
                ps_env=env,
                max_number_of_steps=max_n_steps,
                n_episodes=N_EPISODES,
                rand_qtab=RAND_QTAB,
                learning_rate=LEARNING_RATE,
                discount_factor=DISCOUNT_FACTOR,
                exploration_rate=EXPLORATION_RATE,
                exploration_decay_rate=EXPLORATION_DECAY_RATE,
                k_s=ACTIONS,
                visualize=VISUALIZE
            ),
            base_folder=subfolder,
            n_repeat=N_REPEAT,
            time_step=t,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=None,
            **kwargs
        )


if __name__ == "__main__":
    import time
    start = time.time()
    det_folder = "./results/ps_det_ampl/"
    stoch_folder = "./results/ps_stoch_ampl/"

    det_env = "experiment_pipeline:JMDetCSPSEnv"
    stoch_env = "experiment_pipeline:JMCSPSStochasticEnv"

    N_REPEAT = 5
    TIME_STEP = 1
    P_REF = 1.2
    LOG_LEVEL = logging.INFO
    VISUALIZE = False
    RAND_QTAB = False
    MAX_NUMBER_OF_STEPS = 200
    N_EPISODES = 100
    LEARNING_RATE = 0.5
    DISCOUNT_FACTOR = 0.6
    EXPLORATION_RATE = 0.5
    EXPLORATION_DECAY_RATE = 0.9
    ACTIONS = [0.1, 0.5, 1, 2, 7]
    P_REF_AMPLITUDE = 0.3
    BASELINE_ACTIONS = [0.5, 1, 2, 3, 4, 5, 6, 7]
    # best_combination_experiment(det_folder, env_entry_point=det_env, t_s=[1, 5],
    #                            experiment_name="step_once_not_consid_p_reff",
    #                            p_reff_amplitude=P_REF_AMPLITUDE,
    #                            p_reff_period=200)
    # best_combination_experiment_tsp(det_folder, env_entry_point=det_env, t_s=[1, 5],
    #                                p_reff_amplitude=P_REF_AMPLITUDE,
    #                                p_reff_period=200)
    # baseline_experiment(det_folder, BASELINE_ACTIONS, env_entry_point=det_env,
    #                    n_episodes=5, max_n_steps=MAX_NUMBER_OF_STEPS, time_step=TIME_STEP,
    #                    p_reff=P_REF, log_level=LOG_LEVEL, p_reff_amplitude=P_REF_AMPLITUDE,
    #                    p_reff_period=201, path="../resources/PS_det_JM_2.fmu")
    baseline_experiment(det_folder, BASELINE_ACTIONS, env_entry_point=det_env,
                        n_episodes=5, max_n_steps=40, time_step=5,
                        p_reff=P_REF, log_level=LOG_LEVEL, p_reff_amplitude=P_REF_AMPLITUDE,
                        p_reff_period=201)

    P_REF = 1.1
    # best_combination_experiment(stoch_folder, env_entry_point=stoch_env, t_s=[1, 5],
    #                            experiment_name="step_once_not_consid_p_reff",
    #                            p_reff_amplitude=P_REF_AMPLITUDE,
    #                            p_reff_period=201)
    # best_combination_experiment_tsp(stoch_folder, env_entry_point=stoch_env, t_s=[1, 5],
    #                                p_reff_amplitude=P_REF_AMPLITUDE,
    #                                p_reff_period=201)
    baseline_experiment(stoch_folder, BASELINE_ACTIONS, env_entry_point=stoch_env,
                        n_episodes=20, max_n_steps=MAX_NUMBER_OF_STEPS, time_step=TIME_STEP,
                        p_reff=P_REF, log_level=LOG_LEVEL, p_reff_amplitude=P_REF_AMPLITUDE,
                        p_reff_period=201)
    baseline_experiment(stoch_folder, BASELINE_ACTIONS, env_entry_point=stoch_env,
                        n_episodes=20, max_n_steps=40, time_step=5,
                        p_reff=P_REF, log_level=LOG_LEVEL, p_reff_amplitude=P_REF_AMPLITUDE,
                        p_reff_period=201)
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
