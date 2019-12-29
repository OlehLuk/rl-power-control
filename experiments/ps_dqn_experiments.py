import logging
import os

from agents.ps_dqn import ps_train_test_dqn
from experiment_pipeline import prepare_experiment, run_ps_agent_experiment_with_result_files, suppress_console


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
    discount_factor={},
    exploration_rate={},
    exploration_decay_rate={},
    exploration_rate_final={},
    k_s={},   
    target_update={},
    buffer_size={},
    batch_size={},
    n_hidden=[fc1:{}, fc2:{}],
    exploration decay every () steps,
    log_level=logging.INFO
    {}
    """.format(experiment_name, param_i_correspondence, N_REPEAT, N_EPISODES, VISUALIZE, MAX_NUMBER_OF_STEPS,
               TIME_STEP, P_REF, DISCOUNT_FACTOR, EXPLORATION_DECAY_RATE, EXPLORATION_RATE, EXPLORATION_RATE_FINAL,
               ACTIONS, TARGET_UPDATE, BUFFER_SIZE, BATCH_SIZE, N_HIDDEN_1, N_HIDDEN_2, EXPL_DECAY_STEP, kwargs)

    return descr


def dqn_target_experiment(base_folder, env_entry_point, ws_s,
                          experiment_name="dqn_target", write_large=True, **kwargs):

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
            agent_train_test_once=lambda env: ps_train_test_dqn(
                ps_env=env,
                max_number_of_steps=MAX_NUMBER_OF_STEPS,
                n_episodes=N_EPISODES,
                discount_factor=DISCOUNT_FACTOR,
                exploration_rate=EXPLORATION_RATE,
                exploration_decay_rate=EXPLORATION_DECAY_RATE,
                k_s=ACTIONS,
                window_size=ws,
                n_test_episodes=N_TEST_EPISODES,
                n_test_steps=N_TEST_STEPS,
                target_update=TARGET_UPDATE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                n_hidden_1=N_HIDDEN_1,
                n_hidden_2=N_HIDDEN_2,
                expl_decay_step=EXPL_DECAY_STEP
            ),
            base_folder=subfolder,
            n_repeat=N_REPEAT,
            time_step=TIME_STEP,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=None,
            write_large=write_large,
            q_learning=False,
            save_agent=lambda agent, path: agent.save(path),
            **kwargs
        )


if __name__ == "__main__":
    import time
    start = time.time()
    stoch_folder = "./results/ps_stoch_adv/dqn"

    stoch_env = "experiment_pipeline:JMCSPSStochasticEnv"

    N_REPEAT = 5
    TIME_STEP = 1
    P_REF = 1.2
    LOG_LEVEL = logging.INFO
    VISUALIZE = False
    MAX_NUMBER_OF_STEPS = 200
    N_EPISODES = 100
    N_TEST_EPISODES = 50
    N_TEST_STEPS = None
    LEARNING_RATE = 0.5
    DISCOUNT_FACTOR = 0.6
    EXPLORATION_RATE = 0.5
    EXPLORATION_DECAY_RATE = 0.9996
    ACTIONS = [0.1, 0.5, 1, 2, 7]
    BASELINE_ACTIONS = [0.5, 1, 2, 3, 4, 5, 6, 7]

    EXPL_DECAY_STEP = 1
    TARGET_UPDATE = 100
    EXPLORATION_RATE_FINAL = 0.05
    BUFFER_SIZE = 100
    BATCH_SIZE = 8
    N_HIDDEN_1 = 32
    N_HIDDEN_2 = 32

    TARGET_UPDATE = 25
    N_HIDDEN_1 = 64
    N_HIDDEN_2 = 64
    BATCH_SIZE = 16
    BUFFER_SIZE = 200
    N_EPISODES = 200
    EXPLORATION_RATE_FINAL = 0.01

    dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1],
                                              experiment_name="dqn_best_long")

    # TARGET_UPDATE = 25
    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                                           experiment_name="dqn_targetupdate_low")
    # TARGET_UPDATE = 400
    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                      experiment_name="dqn_targetupdate_high")

    # BUFFER_SIZE = 500
    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                      experiment_name="dqn_expl_big_buffer")

    # BUFFER_SIZE = 100
    # N_HIDDEN_1 = 64
    # N_HIDDEN_2 = 64
    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                       experiment_name="dqn_expl_wide_hidden")

    # N_HIDDEN_1 = 32
    # N_HIDDEN_2 = 32
    # DISCOUNT_FACTOR = 0.9
    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                       experiment_name="dqn_expl_high_discount")

    # DISCOUNT_FACTOR = 0.6
    # BATCH_SIZE = 32
    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                       experiment_name="dqn_expl_big_batch")

    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[1, 2],
    #                      experiment_name="dqn_expl_winsize")

    # BUFFER_SIZE = 50
    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                       experiment_name="dqn_expl_buffer_size")
    # BUFFER_SIZE = 100

    # EXPL_DECAY_STEP = 10
    # EXPLORATION_DECAY_RATE = 0.9
    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                      experiment_name="dqn_expl_expl_step_decay")

    # with suppress_console():

    # SKIP_SECONDS = 175

    # dqn_target_experiment(stoch_folder, env_entry_point=stoch_env, ws_s=[4],
    #                        experiment_name="dqn_skip175",
    #                        simulation_start_time=SKIP_SECONDS)
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
