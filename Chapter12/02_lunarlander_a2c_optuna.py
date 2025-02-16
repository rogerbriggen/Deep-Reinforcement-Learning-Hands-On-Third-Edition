#!/usr/bin/env python3
import gymnasium as gym
import ptan
from ptan.experience import VectorExperienceSourceFirstLast
from ptan.common.utils import TBMeanTracker
import numpy as np
import argparse
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from lib import common
import optuna

MAX_EPISODES = 2000  # Maximum number of episodes per trial

class LunarA2C(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(LunarA2C, self).__init__()

        net_out_size = 512
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, net_out_size),
        )

        self.policy = nn.Sequential(
            nn.Linear(net_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(net_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_out = self.net(x)
        return self.policy(net_out), self.value(net_out)


def objective(trial):

    """ 
    MAX_EPISODES = 800
    num_envs = trial.suggest_int("num_envs", 1, 50)
    Best trial:
    Value:  27.20257568359375
    Params:
        gamma: 0.99506144896081
        learning_rate: 5.861642093052056e-05
        entropy_beta: 0.2102818142977461
        batch_size: 126
        num_envs: 1
        reward_steps: 8
        clip_grad: 0.05084669482248583

    Value: -33
    Params:
        gamma 0.9805805527161346
        learning_rate 0.0021693671242449257
        entropy_beta 0.35022187470044525
        batch_size 26
        num_envs 33
        reward_steps 8
        clip_grad 0.8109772205601864
    """
    """ 
    MAX_EPISODES = 2000
    num_envs = trial.suggest_int("num_envs", 30, 50)
    Best trial:
    Value:  -66.74520111083984
    Params:
        gamma: 0.9494307326016157
        learning_rate: 2.4075605858523512e-05
        entropy_beta: 0.4847935797330713
        batch_size: 46
        num_envs: 32
        reward_steps: 2
        clip_grad: 0.20851543889270752
        eps: 0.001736830440543104
    """


    # Hyperparameters to optimize
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    entropy_beta = trial.suggest_float("entropy_beta", 0.01, 0.5)
    batch_size = trial.suggest_int("batch_size", 8, 128)
    num_envs = trial.suggest_int("num_envs", 30, 50)
    reward_steps = trial.suggest_int("reward_steps", 1, 10)
    clip_grad = trial.suggest_float("clip_grad", 0.01, 1.0)
    eps = trial.suggest_float("eps", 1e-8, 1e-2, log=True)

    env_factories = [
        lambda: gym.make("LunarLander-v2")
        for _ in range(num_envs)
    ]
    env = gym.vector.SyncVectorEnv(env_factories)
    writer = SummaryWriter(comment="-lunarlander-a2c_optuna")

    net = LunarA2C(env.single_observation_space.shape[0],
                   env.single_action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = VectorExperienceSourceFirstLast(
        env, agent, gamma=gamma, steps_count=reward_steps)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=eps)

    batch = []
    done_episodes = 0

    with common.RewardTracker(writer, stop_reward=150) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    done_episodes += 1
                    if tracker.reward(new_rewards[0], step_idx):
                        break
                    if done_episodes >= MAX_EPISODES:
                        print(f"Reached maximum episodes: {MAX_EPISODES}")
                        break

                if len(batch) < batch_size:
                    continue

                states_t, actions_t, vals_ref_t = common.unpack_batch(
                    batch, net, device=device, gamma=gamma, reward_steps=reward_steps)
                batch.clear()

                optimizer.zero_grad()
                logits_t, value_t = net(states_t)
                loss_value_t = F.mse_loss(value_t.squeeze(-1), vals_ref_t)

                log_prob_t = F.log_softmax(logits_t, dim=1)
                adv_t = vals_ref_t - value_t.detach()
                log_act_t = log_prob_t[range(batch_size), actions_t]
                log_prob_actions_t = adv_t * log_act_t
                loss_policy_t = -log_prob_actions_t.mean()

                prob_t = F.softmax(logits_t, dim=1)
                entropy_loss_t = entropy_beta * (prob_t * log_prob_t).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_t.backward(retain_graph=True)
                grads = np.concatenate([
                    p.grad.data.cpu().numpy().flatten()
                    for p in net.parameters() if p.grad is not None
                ])

                # apply entropy and value gradients
                loss_v = entropy_loss_t + loss_value_t
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), clip_grad)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_t

                tb_tracker.track("advantage", adv_t, step_idx)
                tb_tracker.track("values", value_t, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_t, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_t, step_idx)
                tb_tracker.track("loss_policy", loss_policy_t, step_idx)
                tb_tracker.track("loss_value", loss_value_t, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)

    writer.close()
    return np.mean(new_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda", help="Device to use, default=cpu")
    parser.add_argument("--use-async", default=False, action='store_true',
                        help="Use async vector env (A3C mode)")
    parser.add_argument("-n", "--name", default="run1", help="Name of the run")
    args = parser.parse_args()
    device = torch.device(args.dev)

    study_name = "12_02_LunarLander_A2C MaxEpisodes 2000_reward150"
    study_storage = "sqlite:///12_02_lunarlander_a2c.db"
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=study_storage, load_if_exists=True)
    study.optimize(objective, n_trials=30)  # Set the number of trials to 10

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")