import logging
import os
from examples import run_baseline_experiment_with_files, run_experiment_with_result_files, \
    prepare_experiment


def baseline_experiment(base_folder, const_actions, env_entry_point):
    experiment_name = "constant_action_variation"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - {}".format(i, k_s) for i, k_s in enumerate(const_actions)])
        f.write("""{}

Changed constant actions in: 
{}

Other experiment parameters were fixed:
    n_episodes=3,
    max_n_steps=200,
    time_step=1,
    p_reff=1.2,
    const_action=a,
    log_level=logging.INFO,
    p_reff_amplitude=0.3,
    p_reff_period=200
    
            """.format(experiment_name, param_i_correspondence))

    for a in const_actions:
        subfolder = "{}/k={}".format(experiment_folder, a)
        os.mkdir(subfolder)
        run_baseline_experiment_with_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_episodes=10,
            max_n_steps=200,
            time_step=1,
            p_reff=1.1,
            const_action=a,
            log_level=logging.INFO,
            p_reff_amplitude=0.3,
            p_reff_period=201
        )


def best_combination_experiment(base_folder, env_entry_point, t_s):
    experiment_name = "step_once_not_consid_p_reff"
    experiment_folder = prepare_experiment(base_folder, experiment_name)

    if experiment_folder is None:
        return

    description_file = "{}/experiment_description.txt".format(experiment_folder)
    with open(description_file, "w") as f:
        param_i_correspondence = "\n".join(["{} - time_step = {}".format(
            i, t) for i, t in enumerate(t_s)])
        f.write("""{}

Changed time step in: 
{}

Other experiment parameters were fixed:
    n_experiments=5,
    n_episodes=200,
    visualize=False,
    max_n_steps=200//time_step, # modelling 400 seconds horizon
    p_reff=1.3,
    rand_qtab=False,
    learning_rate=0.5,
    discount_factor=0.6,
    k_s=[0.1, 0.5, 1, 2, 7],
    exploration_rate=0.5,
    exploration_decay_rate=0.9,   
    log_level=logging.INFO,
    p_reff_amplitude=0.3,
    p_reff_period=200
        """.format(experiment_name, param_i_correspondence))

    for t in t_s:
        subfolder = "{}/time_step={}".format(experiment_folder, t)
        os.mkdir(subfolder)
        max_n_steps = 200 // t
        run_experiment_with_result_files(
            env_entry_point=env_entry_point,
            base_folder=subfolder,
            n_experiments=5,
            n_episodes=100,
            visualize=False,
            max_n_steps=max_n_steps,
            time_step=t,
            p_reff=1.2,
            rand_qtab=False,
            learning_rate=0.5,
            discount_factor=0.6,
            exploration_rate=0.5,
            exploration_decay_rate=0.9,
            k_s=[0.1, 0.5, 1, 2, 7],
            log_level=logging.INFO,
            p_reff_amplitude=0.3,
            p_reff_period=10,
            path="../resources/jmodelica/linux/PS_ampl_det.fmu"
        )


if __name__ == "__main__":
    import time
    start = time.time()
    folder = "ampl_stoch_ps_exp_results"
    env_entry_point = "examples:JModelicaCSPSEnv"
    # dym_env_class = "examples:DymCSConfigurablePSEnv"
    stoch_env = "examples:JMCSPSStochasticEnv"

    # best_combination_experiment(folder, env_entry_point=env_entry_point, t_s=[1, 5])

    baseline_experiment(folder, [0.5, 1, 2, 5], stoch_env)
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
