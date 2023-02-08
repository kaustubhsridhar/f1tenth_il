import gym
import torch
import numpy as np

import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils

from dataset import Dataset

from bc import bc

from pathlib import Path

def dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, vgain_scales=[1.0]):
    # TODO 
    return 