import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.autograd as Variable
import torch.optim as optim
import random
import copy
from utils import *


import numpy as np
#
class PPOAgent():
    def __init__(self, model, env, env_info, brain, num_agents, device):
        self.num_agents = num_agents
        self.rollout_len = 256
        self.mini_batch_size = 32
        self.epochs = int(self.rollout_len//self.mini_batch_size)
        self.ppo_ratio_clip =.2
        self.entropy_weight = 0.01
        self.gradient_clip = .5
        self.model = model
        self.env = env
        self.brain_name = brain
        self.env_info = env_info
        self.episode_rewards = []
        self.online_rewards = np.zeros(num_agents)
        self.GAMMA = .99
        self.TAU = .95
        self.use_gae = True
        self.total_steps = 0
        self.device = device
        self.debug_messages = True
        self.last_len_rewards = 0
        self.state = self.env_reset()
        self.steps = 0
        self.step_rewards = []
        self.lr_decay = 0.995
        self.lr = 2e-4

    def step(self):

        self.steps +=1

        self.model.update_lr(self.lr_decay**self.steps*self.lr)
        # print('lr',self.lr_decay**self.steps*self.lr)

        trajectory_raw = []

        # states_tensor = self.tensor(self.env_info.vector_observations)
        # Gather training data
        for i in range(self.rollout_len):
            state = self.tensor(self.state)
            # print(state.shape)
            action, log_p, _, value = self.model(state)
            log_p = log_p.detach().cpu().numpy()
            value = value.detach().squeeze(1).cpu().numpy()
            action = action.detach().cpu().numpy()
            next_state, reward, done = self.env_step(action)
            self.online_rewards += reward

            for j, d in enumerate(done):
                if d:
                    self.episode_rewards.append(self.online_rewards[j])
                    self.online_rewards[j] = 0
            trajectory_raw.append((state, action, reward, log_p, value, 1-done))
            self.state = next_state

        next_value = self.model(self.tensor(self.state))[-1].detach().squeeze(1)
        trajectory_raw.append((next_state, action, reward, log_p, value, 1-done))
        trajectory = [None] * (len(trajectory_raw)-1)

        advantages = torch.zeros(self.num_agents,1).to(self.device)
        R =  next_value

        for i in reversed(range(len(trajectory_raw)-1)):

            states, actions, rewards, log_probs, values, not_dones = trajectory_raw[i]
            actions, rewards, not_dones, values, next_values, log_probs = map(
                lambda x: torch.tensor(x).float().to(self.device),
                (actions, rewards, not_dones, values, trajectory_raw[i+1][-2], log_probs))

            R = rewards + self.GAMMA * R * not_dones
            # print('R size:', R.size())
            if not self.use_gae:
                # print('R, values size', R.size(), values_traj[i].detach().size())
                advantages = R[:,None] - values[:,None]
            else:
                td_errors = rewards + self.GAMMA * not_dones * next_values - values
                advantages = advantages * self.TAU * self.GAMMA * not_dones[:,None] + td_errors[:,None]
            trajectory[i] = (states, actions, log_probs, R, advantages)

        states, actions, old_log_probs, returns, advantages, = map(
            lambda x: torch.cat(x, dim=0), zip(*trajectory)
        )
        print('epoch:')
        for i in range(self.epochs):
            print(i, end=' ', flush=True)

            for states_b, actions_b, old_log_probs_b, returns_b, advantages_b in \
                self.get_batch(states, actions, old_log_probs, returns, advantages):

                _, new_log_probs_b, entropy_b, values_b = self.model(states_b, actions_b)

                ratio = (new_log_probs_b - old_log_probs_b).exp()

                clip = torch.clamp(ratio, 1-self.ppo_ratio_clip, 1+self.ppo_ratio_clip)
                clipped_surrogate = torch.min(ratio*advantages_b.unsqueeze(1), clip*advantages_b.unsqueeze(1))

                policy_loss = -torch.mean(clipped_surrogate) - self.entropy_weight * entropy_b.mean()
                value_loss = F.smooth_l1_loss(values_b, returns_b.unsqueeze(1))
                # print(policy_loss.shape, value_loss.shape)
                self.model.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.model.optimizer.step()

        steps = self.rollout_len * self.num_agents
        self.total_steps += steps
        print('\n')

        temp = self.episode_rewards[-100:]
        temp = np.nan_to_num(temp)
        temp = np.mean(temp)
        self.step_rewards.append(temp)

        x = np.arange(0,len(self.episode_rewards[-10:]))
        y = np.nan_to_num(self.episode_rewards[-10:])          #getting occasional NAN.  Not sure why.
        z = np.polyfit(x,y,1)                                   # outputs [a, b] as in ax+b I think
        slope = z[0]

        print('step:{}  act:{:.2f} mean:{:.2f} slope:{:.2f} std:{:.2f}'\
              .format(self.steps,
                      self.step_rewards[-1],
                      np.mean(self.step_rewards[-10:]),
                      slope,
                      np.std(self.step_rewards[-10:])))

        return self.step_rewards

    def tensor(self,x):
        if isinstance(x, torch.Tensor):
            return x
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return x

    def env_step(self,action):
        env_info = self.env.step(action)[self.brain_name]  # get return from environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards
        dones = env_info.local_done
        for i in range(len(dones)):
            dones[i]= int(dones[i])

        return next_states,rewards, self.tensor(dones)

    def env_reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations

    def get_batch(self, states, actions, old_log_probs, returns, advs):
        length = states.shape[0] # nsteps * num_agents
        batch_size = int(length / self.mini_batch_size)
        idx = np.random.permutation(length)
        for i in range(self.mini_batch_size):
            rge = idx[i*batch_size:(i+1)*batch_size]
            yield (
                states[rge], actions[rge], old_log_probs[rge], returns[rge], advs[rge].squeeze(1)
                )
