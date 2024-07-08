"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.Spread3d import Spread3dAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin_target')  # 'kin' or 'rgb' or 'kin_target'
DEFAULT_ACT = ActionType('pid')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = True  # True or False


def evaluate(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO, local=True):
    filename = 'E:\PyProjects\gym-pybullet-drones\gym_pybullet_drones\examples\\results\save-04.26.2024_11.41.42'

    #### Print training progression ############################
    with np.load(filename + '/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j]) + "," + str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename + '/best_model.zip'):
        path = filename + '/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = Spread3dAviary(gui=gui,
                                  obs=DEFAULT_OBS,
                                  act=DEFAULT_ACT,
                                  record=record_video,
                                  need_target=True)
        test_env_nogui = Spread3dAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, need_target=True)
    else:
        test_env = Spread3dAviary(gui=gui,
                                  num_drones=DEFAULT_AGENTS,
                                  obs=DEFAULT_OBS,
                                  act=DEFAULT_ACT,
                                  record=record_video,
                                  need_target=True)
        test_env_nogui = Spread3dAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, need_target=True)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder=output_folder,
                    colab=colab
                    )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:",
              truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                           timestamp=i / test_env.CTRL_FREQ,
                           state=np.hstack([obs2[0:3],
                                            np.zeros(4),
                                            obs2[3:15],
                                            act2
                                            ]),
                           control=np.zeros(12)
                           )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                               timestamp=i / test_env.CTRL_FREQ,
                               state=np.hstack([obs2[d][0:3],
                                                np.zeros(4),
                                                obs2[d][3:15],
                                                act2[d]
                                                ]),
                               control=np.zeros(12)
                               )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool,
                        help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    evaluate(**vars(ARGS))
