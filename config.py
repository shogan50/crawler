import torch
class ConfigPPO():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = 10
        self.gamma = .95
        self.max_steps = int(1e7)
        self.mini_batch_size = 10
        self.lr = 1e-3
        self.tau = .9
        self.fc1_units = 512
        self.fc2_units = 256

class ConfigDDPG:
    def __init__(self,device):
        # Network variables
        self.fc1_units = 512
        self.fc2_units = 256

        # Agent variables
        self.beta = 0.01  # entropy weight
        self.gamma = .99
        self.gae_tau = .95
        self.gradient_clip = .5
        self.lr = 2e-4
        self.lr_decay = 0.995
        self.mini_batch_size = 32
        self.ppo_ratio_clip = .2
        self.rollout_len = 512

        self.use_gae = True

        self.n_epochs = int(self.rollout_len//self.mini_batch_size)

        # other
        self.device = device








