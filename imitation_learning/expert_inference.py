import gym
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os, json
import argparse
import yaml 
import time
import pickle 

from policies.experts.expert_waypoint_follower import ExpertWaypointFollower

import utils.agent_utils as agent_utils
import utils.downsampling as downsampling
import utils.env_utils as env_utils

from policies.agents.agent_mlp import AgentPolicyMLP

def process_parsed_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--training_config', default='il_config.yaml', help='the yaml file containing the training configuration')
    # arg_parser.add_argument('--vgain_scale', type=float, default=1, help='voltage gain scaler (increases top speed)')
    # arg_parser.add_argument('--map_config_location', type=str, help='path to the map config')    
    return arg_parser.parse_args()

def expert_inference(map_config_location, vgain_scale):
    # get args
    parsed_args = process_parsed_args()

    il_config = yaml.load(open(parsed_args.training_config), Loader=yaml.FullLoader)
    il_config['environment']['map_config_location'] = map_config_location # parsed_args.map_config_location
    map_name = il_config['environment']['map_config_location'].split('/')[-1].split('.yaml')[0]

    # set controller params
    tlad = 0.82461887897713965 # look ahead distance
    vgain = 0.90338203837889 * vgain_scale # parsed_args.vgain_scale

    seed = il_config['random_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observation_shape = il_config['policy_type']['agent']['observation_shape']
    downsampling_method = il_config['policy_type']['agent']['downsample_method']

    # Initialize the environment
    map_conf = None
    if il_config['environment']['random_generation'] == False:
        if il_config['environment']['map_config_location'] == None:
            # If no environment is specified but random generation is off, use the default gym environment
            with open('map/example_map/config_example_map.yaml') as file:
                map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        else:
            # If an environment is specified and random generation is off, use the specified environment
            with open(il_config['environment']['map_config_location']) as file:
                map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        map_conf = argparse.Namespace(**map_conf_dict)
        env = gym.make('f110_gym:f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
        env.add_render_callback(env_utils.render_callback)
    else:
        # TODO: If random generation is on, generate random environment
        pass

    # Initialize the expert
    if il_config['policy_type']['expert']['behavior']  == 'waypoint_follower':
        expert = ExpertWaypointFollower(map_conf)
    else:
        # TODO: Implement other expert behavior (Lane switcher and hybrid)
        pass

    # env reset
    traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "rewards": []}
    start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])
    obs, step_reward, done, info = env.reset(start_pose)

    # render options
    render = il_config['environment']['render']
    render_mode = il_config['environment']['render_mode']
    if render:
        if env.renderer is None:
            env.render()

    laptime = 0.0
    start = time.time()
    while not done:
        traj["observs"].append(obs)
        traj["poses_x"].append(obs["poses_x"][0])
        traj["poses_y"].append(obs["poses_y"][0])
        traj["poses_theta"].append(obs["poses_theta"][0])
        # expert action
        scan = agent_utils.downsample_and_extract_lidar(obs, observation_shape, downsampling_method)
        traj["scans"].append(scan)

        # # Add Sim2Real noise
        # sim2real_noise = np.random.uniform(-0.25, 0.25, scan.shape)
        # scan = scan + sim2real_noise

        speed, steer = expert.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], tlad, vgain)
        action = np.array([[steer, speed]])

        # step
        obs, step_reward, done, info = env.step(action)

        traj["actions"].append(action)
        traj["rewards"].append(step_reward)
        # Update rendering
        if render:
            env.render(mode=render_mode)

        print("step_reward: ", step_reward)
        laptime += step_reward
    
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
    os.makedirs('logs', exist_ok=True)
    with open(f"logs/{map_name}_expert_{il_config['policy_type']['expert']['behavior']}_with_vgain_scale_{vgain_scale}.pkl", 'wb') as f:
        pickle.dump(traj, f)

if __name__ == "__main__":
    configs = [
        'map/example_map/config_example_map.yaml',
        'map/levine2nd/levine2nd_config.yaml',
        # 
        # 'map/OG_maps/berlin.yaml',
        # 'map/OG_maps/vegas.yaml',
        # 'map/OG_maps/levine.yaml',
        # 'map/OG_maps/skirk.yaml',
        # 'map/OG_maps/strata_basement.yaml',
    ]
    for file_loc in configs:
        for vg in [0.5, 1.0, 1.5]:
            expert_inference(file_loc, vg)