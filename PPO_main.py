
from config import ConfigPPO
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from PPO_agent import PPOAgent
from PPO_Network import ActorCritic
from unityagents import UnityEnvironment
import torch

cwd = os.getcwd()           # Current Working Directory
if os.name == 'nt':         # 'nt' is what my laptop reports
    cwd = cwd + os.sep + "Crawler_Windows_x86_64"
    env = UnityEnvironment(file_name=cwd + os.sep + "Crawler.exe")
else:                       # assume AWS headless
    cwd = cwd + os.sep + "Crawler_Linux_NoVis"
    env = UnityEnvironment(file_name=cwd + os.sep + "Crawler.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
states = env_info.vector_observations

num_agents = len(env_info.agents)  # number of agents
print('Number of agents:', num_agents)

n_actions = brain.vector_action_space_size
n_inputs = states.shape[1]
config = ConfigPPO()
plot = Plot_Scores(fn='PPO_trial_', text=config.__dict__)

model = ActorCritic(n_inputs, n_actions,config).to(config.device)
agent = PPOAgent(model=model, env=env, env_info=env_info, brain=brain_name,num_agents=num_agents, config=config)
print(model)
last_score = 0
finished_episodes = 0
for step in range(config.max_steps):
    scores_hist = agent.step()
    if scores_hist[-1] > last_score:
        torch.save(model.state_dict(),'crawler.pth')
        last_score = scores_hist[-1]
    plot.plot(scores_hist)


