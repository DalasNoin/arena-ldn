import ppo
import torch
import numpy as np
from torch.distributions.categorical import Categorical

args = ppo.ppo_parse_args()
args.env_id = 'procgen:procgen-starpilot-v0'
args.track = True
# args.capture_video = True
args.total_timesteps = 100_000
args.ent_coef = 0.02
agent = ppo.train_ppo(args)

agent.eval()

#%%

import gym3
import gym.wrappers

# env = gym.make("procgen:procgen-coinrun-v0", render_mode="rgb_array")

from procgen import ProcgenGym3Env
env = ProcgenGym3Env(num=1, env_name="starpilot", render_mode="rgb_array")

env = gym3.VideoRecorderWrapper(env, info_key="rgb", directory="videos-procgen")


done = False
act = gym3.types_np.sample(env.ac_space, bshape=(env.num,))
for i in range(300):
    env.act(act)
    reward, obs, first = env.observe()
    logits = agent.actor(torch.tensor(obs["rgb"]))
    probs = Categorical(logits=logits)
    action = probs.sample().detach().numpy()
 #%%