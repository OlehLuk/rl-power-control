# rl-power-control
### Reinforcement Learning for Energy Imbalance Management using Voltage Control on TCLs

Q-learning was applied to learn an optimal control strategy for Voltage controller in point of common coupling of set of TCL's.
Control goal is to reduce MSE between 

This repository contains:
1. Code for Power System model FMU integration using [ModelicaGym](https://github.com/ucuapps/modelicagym) toolbox.
2. Experiment pipeline.
3. Experiments procedure, corresponding results and their analysis:
  * Q-learning applied to deterministic case with constant reference power - [results](https://github.com/OlehLuk/rl-power-control/blob/master/experiments/results/ps_det_exp_res.ipynb).
  * Q-learning applied to deterministic case with step down in reference power in the middle of the considered interval - [results](https://github.com/OlehLuk/rl-power-control/blob/master/experiments/results/ps_det_ampl_exp_res.ipynb).
  * Q-learning applied to stochastic case with constant reference power - [results](https://github.com/OlehLuk/rl-power-control/blob/master/experiments/results/ps_stoch_exp_res.ipynb).
  * Q-learning applied to stochastic case with step down in reference power in the middle of the considered interval - [results](https://github.com/OlehLuk/rl-power-control/blob/master/experiments/results/ps_stoch_ampl_exp_res.ipynb).

## With support of:
<img src="support.png" alt="supported by UCU.APPS, Eleks, ModelicaGym" width="700" />
