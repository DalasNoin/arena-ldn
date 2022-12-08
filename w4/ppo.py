import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import os
import random
import time
import sys
from distutils.util import strtobool
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch as t
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete
from typing import Any, List, Optional, Union, Tuple, Iterable
from einops import rearrange
from rl_utils import ppo_parse_args, make_env
import tests
# import part4_dqn_solution

MAIN = __name__ == "__main__"
RUNNING_FROM_FILE = True

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class BaseAgent(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.obs_shape = envs.single_observation_space.shape[0]
        self.num_actions = envs.single_action_space.n
        self.hidden_size = 64

class Agent(BaseAgent):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__(envs = envs)

        # todo init layers
        final_critic_linear = nn.Linear(self.hidden_size,1)
        layer_init(final_critic_linear, std=1)
        final_actor_linear = nn.Linear(self.hidden_size,self.num_actions)
        layer_init(final_actor_linear, std=0.01)

        self.critic = nn.Sequential(nn.Linear(self.obs_shape, self.hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(self.hidden_size,self.hidden_size),
                                        nn.Tanh(),
                                        final_critic_linear)
        self.actor = nn.Sequential(nn.Linear(self.obs_shape, self.hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.Tanh(),
                                        final_actor_linear)

class ConvAgent(BaseAgent):

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__(envs = envs)
        # todo init layers
        self.final_critic_linear = nn.Linear(self.hidden_size,1)
        layer_init(self.final_critic_linear, std=1)
        self.final_actor_linear = nn.Linear(self.hidden_size,self.num_actions)
        layer_init(self.final_actor_linear, std=0.01)
        self.norm_constant = 256

        self.actor_critic = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                                        nn.GELU(),
                                        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                                        nn.GELU(),
                                        nn.Conv2d(in_channels=32, out_channels=self.hidden_size, kernel_size=3, stride=2, padding=1),
                                        nn.GELU())

    def actor(self, x):
        x = rearrange(x, "b h w c -> b c h w") /self.norm_constant
        x = self.actor_critic(x)
        x = torch.amax(x, dim=(2,3))
        x = self.final_actor_linear(x)
        return x

    def critic(self, x):
        x = rearrange(x, "b h w c -> b c h w") /self.norm_constant
        x = self.actor_critic(x)
        x = torch.amax(x, dim=(2,3))
        x = self.final_critic_linear(x)
        return x

@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.
    
    next_value: shape (1, env) - represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,) # log prob of the state the model wanted to transition into
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)

    Return: shape (t, env)
    '''
    next_values = t.zeros_like(values)
    next_dones = t.zeros_like(dones)
    advantages = t.zeros_like(dones)
    next_values = t.cat((values[1:], next_value), dim=0)
    next_dones = t.cat((dones[1:], next_done[None]), dim=0)
    # shift values and dones by one and concat next value and next done
    deltas = rewards + gamma*next_values*(1.0 - next_dones)-values
    time_len = advantages.shape[0]
    # calculate the advantages from the deltas
    last_advantage = t.zeros_like(next_done)
    last_done = next_done
    for i in range(time_len-1, -1, -1):
        last_advantage = deltas[i] + gae_lambda * gamma * (1-last_done) * last_advantage
        advantages[i] = last_advantage
        last_done = dones[i]
    return advantages

if MAIN and RUNNING_FROM_FILE:
    tests.test_compute_advantages(compute_advantages)

@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

def minibatch_indexes(batch_size: int, minibatch_size: int, shuffle:bool=False) -> list[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    
    assert batch_size % minibatch_size == 0
    length = batch_size//minibatch_size
    indices = np.arange(batch_size, dtype=np.int32)
    if shuffle:
        np.random.shuffle(indices)
    indices = list(indices.reshape((length,minibatch_size)))

    return indices

if MAIN and RUNNING_FROM_FILE:
    tests.test_minibatch_indexes(minibatch_indexes)

def make_minibatches(
    obs: t.Tensor,
    logprobs: t.Tensor,
    actions: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    obs_shape: tuple,
    action_shape: tuple,
    batch_size: int,
    minibatch_size: int,
) -> list[Minibatch]:
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''
    indices = minibatch_indexes(batch_size=batch_size, minibatch_size=minibatch_size, shuffle=True)
    returns = advantages + values
    minibatches = list()
    for idx in indices:
        minibatches.append(Minibatch(obs=obs.reshape((-1,*obs_shape))[idx], 
                logprobs=logprobs.flatten()[idx],
                actions=actions.reshape((-1,*action_shape))[idx],
                advantages=(advantages).flatten()[idx],
                returns=(returns).flatten()[idx],
                values=(values).flatten()[idx]))
    return minibatches



def calc_policy_loss(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the policy loss, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    mb_action: action that was actually taken

    mb_logprobs: old policy outputs

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.
    '''
    # r denotes the probability ratio between the current policy and the old policy
    num_actions = mb_action.shape[0]
    idx = t.arange(num_actions)
    mb_action = mb_action.to(t.long)
    log_r = probs.logits[idx, mb_action] - mb_logprobs
    r = t.exp(log_r)
    # normalize
    mb_advantages = mb_advantages - t.mean(mb_advantages)
    mb_advantages /= t.std(mb_advantages)

    # mean(min(r*A, clip(r,1-eps,1+eps)A))
    return t.mean(t.min(r*mb_advantages, t.clip(r,1-clip_coef,1+clip_coef)*mb_advantages))



if MAIN and RUNNING_FROM_FILE:
    tests.test_calc_policy_loss(calc_policy_loss)

def calc_value_function_loss(critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    # returns = advantages + values

    # 1/2 the mean squared difference between the critic's prediction and the observed returns
    # from the paper 1/2*(V_theta-V_targ)**2
    V_theta = critic(mb_obs)
    return t.mean(1/2*(V_theta-mb_returns)**2) * vf_coef


if MAIN and RUNNING_FROM_FILE:
    tests.test_calc_value_function_loss(calc_value_function_loss)

def calc_entropy_loss(probs: Categorical, ent_coef: float):
    '''Return the entropy loss term.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return probs.entropy().mean() * ent_coef

if MAIN and RUNNING_FROM_FILE:
    tests.test_calc_entropy_loss(calc_entropy_loss)

class PPOScheduler:
    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.'''
        self.n_step_calls += 1
        
        r = self.n_step_calls / self.num_updates
        lr = self.initial_lr + r * (self.end_lr - self.initial_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

def make_optimizer(agent: BaseAgent, num_updates: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    adam = optim.Adam(agent.parameters(), lr=initial_lr, maximize=True)
    scheduler = PPOScheduler(adam, initial_lr=initial_lr, end_lr=end_lr, num_updates=num_updates)
    return adam, scheduler

@dataclass
class PPOArgs:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

def train_ppo(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = run_name.replace(":", "-")
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for (key, value) in vars(args).items()]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    if "procgen" in args.env_id:
        agent = ConvAgent(envs).to(device)
    else:
        agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    old_approx_kl = 0.0
    approx_kl = 0.0
    value_loss = t.tensor(0.0)
    policy_loss = t.tensor(0.0)
    entropy_loss = t.tensor(0.0)
    clipfracs = []
    info = []
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    for _ in range(num_updates):
        for i in range(0, args.num_steps):
            global_step += args.num_envs

            "(1) YOUR CODE: Rollout phase (see detail #1)"
            obs[i] = next_obs
            dones[i] = next_done
            
            with t.inference_mode():
                V_theta = agent.critic(next_obs)
                logits = agent.actor(next_obs)
                probs = Categorical(logits=logits)
            action = probs.sample()
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            
            rewards[i] = t.from_numpy(reward)
            actions[i] = action
            logprobs[i] = probs.log_prob(action)
            values[i] = V_theta.flatten()

            next_obs = t.tensor(next_obs)
            next_done = t.tensor(done, dtype=t.float32)

            for item in info:
                if isinstance(item,dict) and "episode" in item.keys():
                    # print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
        next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, device, args.gamma, args.gae_lambda)
        clipfracs.clear()
        for _ in range(args.update_epochs):
            minibatches = make_minibatches(
                obs,
                logprobs,
                actions,
                advantages,
                values,
                envs.single_observation_space.shape,
                action_shape,
                args.batch_size,
                args.minibatch_size,
            )
            for mb in minibatches:
                "YOUR CODE: compute loss on the minibatch and step the optimizer (not the scheduler). Do detail #11 (global gradient clipping) here using nn.utils.clip_grad_norm_."
                
                    # obs: t.Tensor
                    # logprobs: t.Tensor
                    # actions: t.Tensor
                    # advantages: t.Tensor
                    # returns: t.Tensor
                    # values: t.Tensor
                
                act_logits = agent.actor(mb.obs)
                probs = Categorical(logits=act_logits)



                policy_loss = calc_policy_loss(probs=probs, mb_action=mb.actions,mb_advantages=mb.advantages, mb_logprobs=mb.logprobs, clip_coef=args.clip_coef)

                value_loss = calc_value_function_loss(agent.critic, mb.obs, mb.returns, vf_coef=args.vf_coef)

                entropy_loss = calc_entropy_loss(probs, ent_coef=args.ent_coef)

                loss = policy_loss - value_loss + entropy_loss

                print(f"loss = {loss.detach().numpy()}, returns={t.mean(mb.returns)}")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        scheduler.step()
        (y_pred, y_true) = (mb.values.cpu().numpy(), mb.returns.cpu().numpy())
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with torch.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            old_approx_kl = (-logratio).mean().item()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/mean_returns", t.mean(mb.returns), global_step)
        if global_step % 10 == 0:
            print("steps per second (SPS):", int(global_step / (time.time() - start_time)))
    envs.close()
    writer.close()
    return agent


from gym.envs.classic_control.cartpole import CartPoleEnv
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
import math

class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, rew, done, info) = super().step(action)
        "YOUR CODE HERE"
        position = obs[0]  #     | 0   | Cart Position         | -4.8                | 4.8               |
        position = np.abs(position /9.6)+0.5
        return obs, rew*position, done, info

gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)

from gym.envs.classic_control.mountain_car import MountainCarEnv

class EasyCar(MountainCarEnv):
    def step(self, action):
        (obs, rew, done, info) = super().step(action)
        "YOUR CODE HERE"
        height = self._height(obs[0])
        velocity = obs[1] * 1e2
        energy = height ** 2 + velocity ** 2

        return obs, energy + rew, done, info

gym.envs.registration.register(id="EasyCar-v0", entry_point=EasyCar, max_episode_steps=500)

if MAIN:
    if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead: python {os.path.basename(filename)} --help")
        args = PPOArgs()
    else:
        args = ppo_parse_args()
        # args.env_id = "EasyCar-v0"
    train_ppo(args)

