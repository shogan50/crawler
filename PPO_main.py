#adapted from https://github.com/rgilman33/baselines-A2C

from config import ConfigPPO
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from PPO_agent import ActorCritic, PPOAgent
from unityagents import UnityEnvironment

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
model = ActorCritic(n_inputs, n_actions).to(config.device)
agent = PPOAgent(model=model, env=env, env_info=env_info, brain=brain_name,num_agents=num_agents, device=config.device)
print(model)

finished_episodes = 0
matplotlib.use('tkagg')  # needed to run on AWS wiht X11 forwarding
line, = plt.plot(np.array(0),np.array(0))
axes = plt.gca()
plt.ion()
plt.xlabel = 'Episode'
plt.ylabel = 'Mean score'
plot = Plot_Scores()

for step in range(config.max_steps):
    scores_hist = agent.step()

    if len(scores_hist) > 2:
        line.set_xdata(np.arange(0, len(scores_hist)))
        line.set_ydata(scores_hist)
        axes.set_xlim(max(0, len(scores_hist) - 2000), len(scores_hist))
        axes.set_ylim(np.min(scores_hist) * 1.05, np.max(scores_hist) * 1.05)
        plt.draw()
        plt.pause(.1)
    #     # line.set_xdata(np.arange(0, len(d_log['returns'][0])))
    #     # line.set_ydata(d_log['returns'][0])
    #     # axes.set_xlim(max(0, len(d_log['returns'][0]) - 2000), len(d_log['returns'][0]))
    #     # axes.set_ylim(np.min(d_log['returns'][0]) * 1.05, np.max(d_log['returns'][0]) * 1.05)
    #     # plt.draw()
    #     # plt.pause(.1)
    #     print(d_log['returns'])
