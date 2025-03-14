#!/usr/bin/env python3
import os
import math
import ptan
import time
import gymnasium as gym
import argparse
from torch.utils.tensorboard.writer import SummaryWriter

from lib import model, common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


GAMMA = 0.99
REWARD_STEPS = 5
BATCH_SIZE = 32
#LEARNING_RATE_ACTOR = 1e-5
#LEARNING_RATE_CRITIC = 1e-3
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-2
ENTROPY_BETA = 1e-3
ENVS_COUNT = 16

TEST_ITERS = 100000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda", help="Device to use, default=cpu")
    parser.add_argument("-n", "--name", default="LunarLanderContWind", help="Name of the run")
    parser.add_argument("-e", "--env", choices=list(common.ENV_PARAMS.keys()),
                        default='lunar', help="Environment id, default=lunar")
    parser.add_argument("--mujoco", default=False, action='store_true',
                        help="If given, MuJoCo env will be used instead of PyBullet")
    parser.add_argument("--no-unhealthy", default=False, action='store_true',
                        help="Disable unhealthy checks in MuJoCo env")
    args = parser.parse_args()
    device = torch.device(args.dev)
    name = args.name + "_v2"
    #name += "-mujoco" if args.mujoco else "-pybullet"

    save_path = os.path.join("saves", "a2c-" + name)
    os.makedirs(save_path, exist_ok=True)

    extra = {}
    if args.mujoco and args.no_unhealthy:
        extra['terminate_when_unhealthy'] = False

    #env_id = common.register_env_lunar(args.env, args.mujoco)
    extra['continuous'] = True
    # Set if you want to train with wind... it makes the training harder
    extra['enable_wind'] = True     # default False

    env_id = "LunarLander-v2"
    envs = []
    for i in range(ENVS_COUNT):
        extra['wind_power'] = i * 2     # default 15.0
        extra['turbulence_power'] = 3.0 # default 1.5
        envs.append(gym.make(env_id, **extra))
    extra['wind_power'] = 15.0     # default 15.0
    extra['turbulence_power'] = 1.5 # default 1.5
    test_env = gym.make(env_id, **extra)

    net_act = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
    net_crt = model.ModelCritic(envs[0].observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment="-a2c_" + name)
    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = model.test_net(net_act, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net_act.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_a2c(batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                opt_act.zero_grad()
                mu_v = net_act(states_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * model.calc_logprob(mu_v, net_act.logstd, actions_v)  #mu_v, net_act.logstd = sigma (varianz), actions_v = choosen action
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)

