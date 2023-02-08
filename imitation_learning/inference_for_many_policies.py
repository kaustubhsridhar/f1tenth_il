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

import utils.downsampling as downsampling
import utils.env_utils as env_utils

from policies.agents.agent_mlp import AgentPolicyMLP

def process_parsed_args():
    arg_parser = argparse.ArgumentParser()
    return arg_parser.parse_args()

def inference(map_loc, training_config, model_path, seed):
    parsed_args = process_parsed_args()

    il_config = yaml.load(open(training_config), Loader=yaml.FullLoader)
    il_config['environment']['map_config_location'] = map_loc

    model_type = il_config['policy_type']['agent']['model']

    seed = il_config['random_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'mlp':
        agent = AgentPolicyMLP(il_config['policy_type']['agent']['observation_shape'], \
                                il_config['policy_type']['agent']['hidden_dim'], \
                                2, \
                                il_config['policy_type']['agent']['learning_rate'], \
                                device)
    else:
        #TODO: Implement other model (Transformer)
        pass

    agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
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
        raw_lidar_scan = obs["scans"][0]
        processed_lidar_scan = downsampling.downsample(raw_lidar_scan, observation_shape, downsampling_method)
        traj["scans"].append(processed_lidar_scan)

        action = agent.get_action(processed_lidar_scan)
        action_expand = np.expand_dims(action, axis=0)
        obs, reward, done, _ = env.step(action_expand)

        print("step_reward: ", step_reward)

        laptime += step_reward

        traj["actions"].append(action)
        traj["rewards"].append(step_reward)
        if render:
            env.render(mode=render_mode)
    
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
    os.makedirs('logs', exist_ok=True)
    model_identifier = model_path.split('/')[-1].split('.')[0]
    with open(f"logs/{model_identifier}.pkl", 'wb') as f:
        pickle.dump(traj, f)

if __name__ == '__main__':
    all_map_locs = [
        # 'map/example_map/config_example_map.yaml',
        'map/levine2nd/levine2nd_config.yaml',
        # 
        # 'map/OG_maps/berlin.yaml',
        # 'map/OG_maps/vegas.yaml',
        # 'map/OG_maps/levine.yaml',
        # 'map/OG_maps/skirk.yaml',
        # 'map/OG_maps/strata_basement.yaml',
    ]
    all_seeds = [0, 1, 2]
    all_algorithms = ['BehavioralCloning']#, 'dagger', 'hg-dagger']

    print(f'----------------------------------------')
    print(f'Unique expert imitation')
    print(f'----------------------------------------')
    for map_loc in all_map_locs:
        map_name = map_loc.split('/')[-1].split('.yaml')[0]
        for algorithm in all_algorithms:
            for type in ['slow', 'normal', 'fast']:
                print(f'********** map {map_loc} with {algorithm} and type {type} ********** (fixed: seed 0)')
                inference(map_loc=map_loc, training_config=f'configs/unique_{type}.yaml', model_path=f'logs/training/{map_name}_{algorithm}_unique_{type}_model.pkl', seed = 0)

    print(f'----------------------------------------')
    print(f'Mixed expert imitation')
    print(f'----------------------------------------')
    for map_loc in all_map_locs:
        map_name = map_loc.split('/')[-1].split('.yaml')[0]
        for algorithm in all_algorithms:
            for s in all_seeds:
                print(f'********** map {map_loc} with {algorithm} and seed {s} (fixed: type normal **********')
                inference(map_loc=map_loc, training_config=f'configs/mixed_seed_{s}.yaml', model_path=f'logs/training/{map_name}_{algorithm}_mixed_seed_{s}_model.pkl', seed = s)
