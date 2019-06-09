from agent_dir.agent import Agent
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import torch.nn as nn
import random
import logging
import os
import sys
import multiprocessing


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class A2C(nn.Module):
    def __init__(self, in_channels=4, action_num=3):

        super(A2C, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(7*7*64, 512),
            nn.ReLU()
        )
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, action_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.main(x / 255.0)
        a = self.actor(x)
        c = self.critic(x)
        return c, a

class ReplayBuffer(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_num):
        self.observation = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, action_num)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0
    def to(self, device):
        self.observation = self.observation.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.observation[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observation[0].copy_(self.observation[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]


class Agent_A2C(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_A2C,self).__init__(env)
        self.args = args
        self.device = self._prepare_gpu()
        self.action_num = 3
        self.discount_factor = 0.99
        self.n_steps = 5
        self.n_cpu = multiprocessing.cpu_count()
        self.n_batch = self.n_steps * self.n_cpu
        self.batch_size = 32
        self.n_frames = 80e6
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.max_grad_norm = 0.5
        self.Actor_Critic = A2C(action_num=self.action_num).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.Actor_Critic.parameters(), lr=7e-4, eps=1e-5)
        self.reward_list = []
        self.log_list = []
        self.save_dir = self.args.save_dir

        os.makedirs(args.save_dir, exist_ok=True)
        handlers = [logging.FileHandler(os.path.join(args.save_dir, 'output.log'), mode = 'w'), logging.StreamHandler()]
        logging.basicConfig(handlers = handlers, level=logging.INFO, format='')
        self.logger = logging.getLogger()

        self.save_dir = self.save_dir + "a2c"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if args.test_a2c:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        checkpoint = torch.load(self.args.check_path)
        self.Actor_Critic.load_state_dict(checkpoint['state_dict'])
        self.Actor_Critic.eval()
        self.Actor_Critic.to(self.device)
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        rollouts = ReplayBuffer(self.n_steps, self.n_cpu,self.env.observation_space.shape, self.action_num)
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
        rollouts.observation[0].copy_(obs)
        rollouts.to(self.device)
        episode_rewards = torch.zeros(self.n_cpu, 1)
        final_rewards = torch.zeros(self.n_cpu, 1)
        for update in range(1, int(self.n_frames//self.n_batch)+1):
            for step in range(self.n_steps):

                with torch.no_grad():
                    value, action_feature = self.Actor_Critic(rollouts.observation[step])
                    prob = F.softmax(action_feature, dim=1)
                    log_prob = F.log_softmax(action_feature, dim=1).data
                    action = prob.multinomial(1).data

                cpu_actions = action.cpu().numpy()
                obs, reward, done, infos = self.env.step(cpu_actions + 1)
                obs = torch.FloatTensor(obs)
                reward = torch.FloatTensor(reward).unsqueeze(1)
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                #print(obs.size(), action.size(), log_prob.size(), value.size(), reward.size(), masks.size())
                rollouts.insert(obs, action, log_prob, value, reward, masks)

                episode_rewards += reward
                final_rewards *= masks[step].cpu()
                final_rewards += (1 - masks[step].cpu()) * episode_rewards
                episode_rewards *= masks[step].cpu()


            with torch.no_grad():
                next_value, _ = self.Actor_Critic(rollouts.observation[-1])
                next_value = next_value.detach()
            rollouts.compute_returns(next_value, self.discount_factor)

            # The following is for A2C
            values, action_features = self.Actor_Critic(rollouts.observation[:-1].view(-1, *self.env.observation_space.shape))
            probs = F.softmax(action_features, dim=1).view(self.n_steps, self.n_cpu, -1)
            log_probs = F.log_softmax(action_features, dim=1).view(self.n_steps, self.n_cpu, -1)
            values = values.view(self.n_steps, self.n_cpu, 1)
            action_log_probs = log_probs.gather(2, rollouts.actions)

            dist_entropy = -(log_probs * probs).sum(-1).mean()

            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(advantages.detach() * action_log_probs).mean()

            self.optimizer.zero_grad()
            (value_loss * self.vf_coef + action_loss - dist_entropy * self.ent_coef).backward()

            nn.utils.clip_grad_norm_(self.Actor_Critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.logger.info("Updates {}, steps {}, reward {:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".format(update, update * self.n_cpu * self.n_steps, final_rewards.mean(), -dist_entropy.item(), value_loss.item(), action_loss.item()))
            self.reward_list.append(final_rewards.mean())
            if update % 1000 == 0:
                log = {
                    'update': update,
                    'loss': value_loss * self.vf_coef + action_loss - dist_entropy * self.ent_coef,
                    'step': update * self.n_cpu * self.n_steps,
                    'latest_reward': self.reward_list
                }
                self.log_list.append(log)
                checkpoint = {
                    'log': self.log_list,
                    'state_dict': self.Actor_Critic.state_dict()
                }
                torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_episode{}.pth.tar'.format(update/1000)))
                self.logger.info('save checkpoint_episode{}.pth.tar!'.format(update/1000))

        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        value, action_feature = self.Actor_Critic((torch.FloatTensor(observation).unsqueeze(0)).to(self.device))
        prob = F.softmax(action_feature, dim=1)
        action = prob.max(-1)[1].data[0]
        return action.item() + 1

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

