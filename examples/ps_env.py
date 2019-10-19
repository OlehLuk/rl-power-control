import logging
import random
import time

import numpy as np
from gym import spaces
from modelicagym.environment import JModCSEnv, DymolaCSEnv

logger = logging.getLogger(__name__)


P_DIFF_THRESHOLD = 0.5


class PSEnv:
    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True if everything worked out.
        """
        return True

    def render(self, mode='human', close=False):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        Draws cart-pole with the built-in gym tools.

        :param mode: rendering mode. Read more in Gym docs.
        :param close: flag if rendering procedure should be finished and resources cleaned.
        Used, when environment is closed.
        :return: rendering result
        """
        u, p = self.state
        print("Current value of u: {}, p_reference: {}".format(u, p))
        return None

    def reset(self):
        """
        OpenAI Gym API. Restarts environment.
        Cleans saved difference.
        :return: state after restart
        """
        self.p_diffs = []
        return super().reset()

    def _get_action_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns action space according to OpenAI Gym API requirements

        :return: Discrete action space of size 7, as currently only 7 actions are considered:
        k = 0,1,2,..,6
        """
        return spaces.Box(np.array([0]), np.array([np.inf]))

    def _get_observation_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns observation space according to OpenAI Gym API requirements

        :return: Box state space with specified lower and upper bounds for state variables.
        """
        high = np.array([np.inf, np.inf])

        return spaces.Box(np.array([0, 0]), high)

    def _is_done(self):
        """
        Internal logic that is utilized by parent classes.
        Checks if power demand is not higher than reference + threshold.

        :return: boolean flag if current state of the environment indicates that experiment has ended.
        True, if current value of demand is bigger than p_reference + threshold
        """
        u, p = self.state
        logger.debug("u (actual): {}".format(u))

        if u > p + self.p_diff_threshold:
            return True
        else:
            return False

    def _reward_policy(self):
        """
        Internal logic that is utilized by parent classes. Provides custom reward policy.
        :return: MSE between vector of reference and actual demand
        """
        u, p = self.state
        if self.compute_reward is None:
            return -1000 * (u - p) * (u - p) if not self._is_done() else -1000
        else:
            return self.compute_reward(u, p)


class JModelicaCSPSEnv(PSEnv, JModCSEnv):
    """
    Class that integrates Power System FMU as an environments for experiments.
    Positive and negative reward were not specified in config for ModelicaBaseEnv,
    as class imlements custom _reward_policy().

    Attributes:
        p_reff (float): p_reference parameter of the system.
        time_step (float): time difference between simulation steps.
        log_level: level of logging to be used
    """
    def __init__(self,
                 p_reff,
                 time_step,
                 log_level,
                 compute_reward=None,
                 p_reff_amplitude=0,
                 p_reff_period=200,
                 path="../resources/jmodelica/linux/PowerSystems_Examples_TCL_ULTC_20_LTC_P_New.fmu"):

        logger.setLevel(log_level)
        self.p_reff = p_reff
        self.p_diff_threshold = P_DIFF_THRESHOLD
        self.compute_reward = compute_reward
        self.p_diffs = []
        self.viewer = None
        self.display = None

        config = {
            'model_input_names': ['k_in'],
            'model_output_names': ['controller_New.u2', 'controller_New.p'],
            'model_parameters': {'P_ref.height': p_reff, 'P_ref_change.amplitude': p_reff_amplitude,
                                 'P_ref_change.period': p_reff_period},
            'time_step': time_step
        }
        super().__init__(path,
                         config, log_level)

    # modelicagym API implementation


class JMCSPSStochasticEnv(PSEnv, JModCSEnv):

    def __init__(self,
                 p_reff,
                 time_step,
                 log_level,
                 compute_reward=None,
                 p_reff_amplitude=0,
                 p_reff_period=200,
                 get_seed=lambda: round(time.time()),
                 p_diff_threshold=P_DIFF_THRESHOLD,
                 path="../resources/jmodelica/linux/PS_stochastic_JM.fmu"):

        logger.setLevel(log_level)
        self.p_reff = p_reff
        self.p_diff_threshold = p_diff_threshold
        self.compute_reward = compute_reward
        self.p_diffs = []
        self.viewer = None
        self.display = None
        self.get_seed = get_seed

        config = {
            'model_input_names': ['k_in'],
            'model_output_names': ['controller_New.u2', 'controller_New.p'],
            'model_parameters': {'P_ref.height': p_reff, 'P_ref_change.amplitude': p_reff_amplitude,
                                 'P_ref_change.period': p_reff_period, "globalSeed.useAutomaticSeed": 0,
                                 "globalSeed.fixedSeed": self.get_seed()},
            'time_step': time_step
        }
        super().__init__(path,
                         config, log_level)

    def reset(self):
        """
        OpenAI Gym API. Restarts environment.
        Cleans saved difference.
        :return: state after restart
        """
        self.model_parameters.update({"globalSeed.useAutomaticSeed": 0})
        self.model_parameters.update({"globalSeed.fixedSeed": self.get_seed()})
        return super().reset()


class DymCSConfigurablePSEnv(PSEnv, DymolaCSEnv):
    """
    Class that integrates Power System FMU as an environments for experiments.
    Positive and negative reward were not specified in config for ModelicaBaseEnv,
    as class imlements custom _reward_policy().

    Attributes:
        p_reff (float): p_reference parameter of the system.
        time_step (float): time difference between simulation steps.
        log_level: level of logging to be used
    """
    def __init__(self,
                 p_reff,
                 time_step,
                 log_level,
                 p_reff_amplitude=0,
                 p_reff_period=200,
                 p_diff_threshold=P_DIFF_THRESHOLD,
                 compute_reward=None,
                 path="../resources/PowerSystems_Examples_TCL_0ULTC_020_0LTC_0P_0New_cvode.fmu"):

        logger.setLevel(log_level)
        self.p_reff = p_reff
        self.p_diff_threshold = p_diff_threshold
        self.compute_reward = compute_reward
        self.p_diffs = []
        self.viewer = None
        self.display = None

        config = {
            'model_input_names': ['k_in'],
            'model_output_names': ['controller_New.u2', 'controller_New.p'],
            'model_parameters': {'P_ref.height': p_reff, 'P_ref_change.amplitude': p_reff_amplitude,
                                 'P_ref_change.period': p_reff_period},
            'time_step': time_step
        }
        super().__init__(path,
                         config, log_level)
