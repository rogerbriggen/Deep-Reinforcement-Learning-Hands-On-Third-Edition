#!/usr/bin/env python3
import argparse
import gymnasium as gym

from lib import common, model, kfac

import numpy as np
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("-m", "--model", default="./saves/a2c-LunarLanderCont/best_+303.849_18200000.dat", help="Model file to load")
    parser.add_argument("-m", "--model", default="./saves/a2c-LunarLanderContWind_v2/best_+288.780_59200000.dat", help="Model file to load")
    parser.add_argument("-e", "--env", choices=list(common.ENV_PARAMS.keys()),
                        default='cheetah', help="Environment name to use, default=cheehah")
    parser.add_argument("-r", "--record", default="vids/a2c-LunarLanderContWind", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("--acktr", default=False, action='store_true', help="Enable Acktr-specific tweaks")
    parser.add_argument("--mujoco", default=False, action='store_true', help="Enable MuJoCo, default=PyBullet")
    args = parser.parse_args()

    #env_id = common.register_env(args.env, args.mujoco)
    extra = {}
    extra['continuous'] = True
    extra['enable_wind'] = True     # default False
    extra['wind_power'] = 15.0      # default 15.0
    extra['turbulence_power'] = 1.5 # default 1.5
    env_id = "LunarLander-v2"
    env = gym.make(env_id, render_mode='rgb_array', **extra)
    if args.record is not None:
        env = gym.wrappers.RecordVideo(env, video_folder=args.record)

    net = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0])
    if args.acktr:
        opt = kfac.KFACOptimizer(net)
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'), weights_only=True))

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor(obs)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        if np.isscalar(action): 
            action = [action]
        obs, reward, done, is_tr, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done or is_tr:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
