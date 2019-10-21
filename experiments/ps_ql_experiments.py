import logging
import os
import sys

from agents import ps_train_test_osp_ql
from experiment_pipeline import prepare_experiment, run_ps_agent_experiment_with_result_files, \
    run_ps_const_control_experiment_with_files, baseline_experiment, suppress_console


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
    log_level=logging.INFO
    """.format(experiment_name, param_i_correspondence, N_REPEAT, N_EPISODES, VISUALIZE, MAX_NUMBER_OF_STEPS,
               TIME_STEP, P_REF, RAND_QTAB, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_DECAY_RATE, EXPLORATION_RATE,
               ACTIONS)

    return descr


def ks_experiment(base_folder, ks, env_entry_point,
                  experiment_name="action_space(k_s)_variation"):

    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "Changed k_s in:" + "\n".join(["{} - {}".format(i, k_s) for i, k_s in enumerate(ks)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for k_s in ks:
        subfolder = "{}/k_s={}".format(experiment_folder, k_s)
        os.mkdir(subfolder)
        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_osp_ql(
                ps_env=env,
                max_number_of_steps=MAX_NUMBER_OF_STEPS,
                n_episodes=N_EPISODES,
                rand_qtab=RAND_QTAB,
                learning_rate=LEARNING_RATE,
                discount_factor=DISCOUNT_FACTOR,
                exploration_rate=EXPLORATION_RATE,
                exploration_decay_rate=EXPLORATION_DECAY_RATE,
                k_s=k_s,
                visualize=VISUALIZE
            ),
            base_folder=subfolder,
            n_repeat=N_REPEAT,
            time_step=TIME_STEP,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=None,
        )


def exploration_experiment(base_folder, explor_params, env_entry_point,
                           experiment_name="exploration(rate_and_discount)_variation"):

    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - exploration rate = {}, exploration rate decay = {}".format(
            i, params[0], params[1]) for i, params in enumerate(explor_params)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for expl_rate, expl_decay in explor_params:
        subfolder = "{}/expl_rate={}_decay={}".format(experiment_folder, expl_rate, expl_decay)
        os.mkdir(subfolder)
        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_osp_ql(
                ps_env=env,
                max_number_of_steps=MAX_NUMBER_OF_STEPS,
                n_episodes=N_EPISODES,
                rand_qtab=RAND_QTAB,
                learning_rate=LEARNING_RATE,
                discount_factor=DISCOUNT_FACTOR,
                exploration_rate=expl_rate,
                exploration_decay_rate=expl_decay,
                k_s=ACTIONS,
                visualize=VISUALIZE
            ),
            base_folder=subfolder,
            n_repeat=N_REPEAT,
            time_step=TIME_STEP,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=None,
        )


def timestep_experiment(base_folder, t_s, env_entry_point):
    experiment_name = "time_step_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - time_step = {}".format(
            i, t) for i, t in enumerate(t_s)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for t in t_s:
        subfolder = "{}/time_step={}".format(experiment_folder, t)
        os.mkdir(subfolder)
        max_n_steps = 200 // t
        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_osp_ql(
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
        )


def learning_rate_experiment(base_folder, lr_s, env_entry_point):
    experiment_name = "learning_rate_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - learning_rate = {}".format(
            i, lr) for i, lr in enumerate(lr_s)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for lr in lr_s:
        subfolder = "{}/learning_rate={}".format(experiment_folder, lr)
        os.mkdir(subfolder)
        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_osp_ql(
                ps_env=env,
                max_number_of_steps=MAX_NUMBER_OF_STEPS,
                n_episodes=N_EPISODES,
                rand_qtab=RAND_QTAB,
                learning_rate=lr,
                discount_factor=DISCOUNT_FACTOR,
                exploration_rate=EXPLORATION_RATE,
                exploration_decay_rate=EXPLORATION_DECAY_RATE,
                k_s=ACTIONS,
                visualize=VISUALIZE
            ),
            base_folder=subfolder,
            n_repeat=N_REPEAT,
            time_step=TIME_STEP,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=None,
        )


def discount_factor_experiment(base_folder, df_s, env_entry_point):
    experiment_name = "discount_factor_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - discount_factor = {}".format(
            i, df) for i, df in enumerate(df_s)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for df in df_s:
        subfolder = "{}/discount_factor={}".format(experiment_folder, df)
        os.mkdir(subfolder)
        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_osp_ql(
                ps_env=env,
                max_number_of_steps=MAX_NUMBER_OF_STEPS,
                n_episodes=N_EPISODES,
                rand_qtab=RAND_QTAB,
                learning_rate=LEARNING_RATE,
                discount_factor=df,
                exploration_rate=EXPLORATION_RATE,
                exploration_decay_rate=EXPLORATION_DECAY_RATE,
                k_s=ACTIONS,
                visualize=VISUALIZE
            ),
            base_folder=subfolder,
            n_repeat=N_REPEAT,
            time_step=TIME_STEP,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=None,
        )


def reward_experiment(base_folder, env_entry_point, compute_reward_s):
    experiment_name = "reward_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)

    param_i_correspondence = "\n".join(["{} - reward = {}".format(
        i, comp_rew[0]) for i, comp_rew in enumerate(compute_reward_s)])

    with open(description_file, "w") as f:
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for comp_rew in compute_reward_s:
        name, func = comp_rew
        subfolder = "{}/reward-{}".format(experiment_folder, name)
        os.mkdir(subfolder)
        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_osp_ql(
                ps_env=env,
                max_number_of_steps=MAX_NUMBER_OF_STEPS,
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
            time_step=TIME_STEP,
            p_reff=P_REF,
            log_level=LOG_LEVEL,
            env_entry_point=env_entry_point,
            compute_reward=func,
        )


def validation_experiment(base_folder, env_entry_point):
    experiment_name = "setup_validation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        f.write(gen_exp_descr(experiment_name, None))
    run_ps_agent_experiment_with_result_files(
        agent_train_test_once=lambda env: ps_train_test_osp_ql(
            ps_env=env,
            max_number_of_steps=MAX_NUMBER_OF_STEPS,
            n_episodes=N_EPISODES,
            rand_qtab=RAND_QTAB,
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
            exploration_rate=EXPLORATION_RATE,
            exploration_decay_rate=EXPLORATION_DECAY_RATE,
            k_s=ACTIONS,
            visualize=VISUALIZE
        ),
        base_folder=experiment_folder,
        n_repeat=N_REPEAT,
        time_step=TIME_STEP,
        p_reff=P_REF,
        log_level=LOG_LEVEL,
        env_entry_point=env_entry_point,
        compute_reward=None,
    )


def best_combination_experiment(base_folder, env_entry_point, t_s,
                                experiment_name="best_parameters_combination", **kwargs):

    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - time_step = {}".format(
            i, t) for i, t in enumerate(t_s)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for t in t_s:
        subfolder = "{}/time_step={}".format(experiment_folder, t)
        os.mkdir(subfolder)
        max_n_steps = MAX_NUMBER_OF_STEPS // t
        run_ps_agent_experiment_with_result_files(
            agent_train_test_once=lambda env: ps_train_test_osp_ql(
                ps_env=env,
                max_number_of_steps=max_n_steps,
                n_episodes=N_EPISODES,
                rand_qtab=RAND_QTAB,
                learning_rate=LEARNING_RATE,
                discount_factor=DISCOUNT_FACTOR,
                exploration_rate=EXPLORATION_RATE,
                exploration_decay_rate=EXPLORATION_DECAY_RATE,
                k_s=ACTIONS,
                visualize=VISUALIZE,
                n_test_episodes=N_TEST_EPISODES
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


def validate_stochastic_experiment(base_folder, const_actions, env_entry_point):
    experiment_name = "stoch-env_validation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - {}".format(i, k_s) for i, k_s in enumerate(const_actions)])
        f.write(gen_exp_descr(experiment_name, param_i_correspondence))

    for a in const_actions:
        subfolder = "{}/k={}".format(experiment_folder, a)
        os.mkdir(subfolder)
        run_ps_const_control_experiment_with_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_episodes=3,
            max_n_steps=200,
            time_step=1,
            p_reff=1.3,
            const_action=a,
            log_level=logging.INFO,
            p_reff_amplitude=0,
            p_reff_period=200,
            get_seed=lambda: 23
        )


if __name__ == "__main__":
    import time
    start = time.time()
    det_folder = "./results/ps_det/"
    stoch_folder = "./results/ps_stoch/"

    det_env = "experiment_pipeline:JMDetCSPSEnv"
    stoch_env = "experiment_pipeline:JMCSPSStochasticEnv"

    N_REPEAT = 5
    TIME_STEP = 1
    P_REF = 1.3
    LOG_LEVEL = logging.INFO
    VISUALIZE = False
    RAND_QTAB = False
    MAX_NUMBER_OF_STEPS = 200
    N_EPISODES = 200
    N_TEST_EPISODES = 50
    LEARNING_RATE = 0.5
    DISCOUNT_FACTOR = 0.6
    EXPLORATION_RATE = 0.5
    EXPLORATION_DECAY_RATE = 0.9
    ACTIONS = [0.1, 0.5, 1, 2, 7]
    BASELINE_ACTIONS = [0.5, 1, 2, 3, 4, 5, 6, 7]
    # following experiments take significant amount of time, so it is advised to run only one of them at once
    # 1
    # ks_experiment(det_folder, ks=[
    #     [1, 2, 3],
    #    [0.5, 1, 2, 3, 4],
    #    [0.1, 0.5, 1, 2, 7]
    # ], env_entry_point=det_env)

    # 2
    # exploration_experiment(det_folder, explor_params=[[1, 0.99],
    #                                              [0.2, 1]], env_entry_point=env_entry_point)

    # 3
    # timestep_experiment(det_folder, [2, 5], env_entry_point=env_entry_point)

    # 4
    # learning_rate_experiment(det_folder, [0.2, 0.8], env_entry_point=env_entry_point)

    # 5
    # discount_factor_experiment(det_folder, [0.2, 0.9], env_entry_point=env_entry_point)

    # 6
    # baseline_experiment(det_folder, BASELINE_ACTIONS, env_entry_point=det_env,
    #                    n_episodes=5, max_n_steps=MAX_NUMBER_OF_STEPS, time_step=TIME_STEP,
    #                    p_reff=P_REF, log_level=LOG_LEVEL)
    # baseline_experiment(det_folder, BASELINE_ACTIONS, env_entry_point=det_env,
    #                    n_episodes=5, max_n_steps=40, time_step=5,
    #                    p_reff=P_REF, log_level=LOG_LEVEL)
    # 7
    # reward_experiment(det_folder, env_entry_point, [
    #    ["-MSEx100", lambda u, p: -100*(u-p)**2],
    #    ["-MAEx100", lambda u, p: -100 * abs(u - p)]
    # ])

    # 8
    # best_combination_experiment(det_folder, env_entry_point=det_env, t_s=[1, 5],
    #                            experiment_name="best_params_longer_train")
    #
    P_REF = 1.2

    with suppress_console():
        best_combination_experiment(stoch_folder, env_entry_point=stoch_env, t_s=[5],
                                    experiment_name="best_params_longer_train")

    # baseline_experiment(stoch_folder, BASELINE_ACTIONS, env_entry_point=stoch_env,
    #                    n_episodes=20, max_n_steps=MAX_NUMBER_OF_STEPS, time_step=TIME_STEP,
    #                    p_reff=P_REF, log_level=LOG_LEVEL)
    # baseline_experiment(stoch_folder, BASELINE_ACTIONS, env_entry_point=stoch_env,
    #                    n_episodes=20, max_n_steps=40, time_step=5,
    #                    p_reff=P_REF, log_level=LOG_LEVEL)

    # validate_stochastic_experiment(stoch_folder, [0.5, 1, 2, 5], stoch_env)
    # validation_experiment(stoch_folder, [0.5, 1, 2, 5], env_entry_point=env_entry_point)

    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
