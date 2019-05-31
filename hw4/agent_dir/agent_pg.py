from agent_dir.agent import Agent
import scipy.misc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import os
import logging
import time


def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size).astype(np.float32)

    return np.expand_dims(resized, axis=2)

class Policy_Gradient(nn.Module):
    def __init__(self):
        super(Policy_Gradient, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(80 * 80 * 1, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, _input):
        _input = _input.reshape(_input.shape[0], -1)
        output = self.FC(_input)
        return output
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(Policy_Gradient, self).__str__() + '\nTrainable parameters: {}'.format(params)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        env.reset()
        self.args = args
        self.model = Policy_Gradient()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-4)
        # self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-4)
        self.criterion = nn.functional.binary_cross_entropy
        self.action_dict = {0: 2, 1: 3} # map 0 to UP, map 1 to DOWN
        self.log_list = []
        self.env.seed(1230)
        self.start_time = time.time()
        self.start_episode = 0
        self.test_last_observation = None
        # open save dir
        os.makedirs(args.save_dir, exist_ok=True)
        # set up logger
        handlers = [logging.FileHandler(os.path.join(args.save_dir, 'output.log'), mode = 'w'), logging.StreamHandler()]
        logging.basicConfig(handlers = handlers, level=logging.INFO, format='')
        self.logger = logging.getLogger()
        # set up device
        n_gpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        # load checkpoint
        if args.check_path != None: self._load_checkpoint()
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        self.model.eval()
        self.model.to(self.device)

    def train(self):
        
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        env = self.env
        model = self.model
        episode_n = self.start_episode
        save_n = int(np.sqrt(self.start_episode)) + 1
        model.train()
        model.to(self.device)

        while True:
            # =================
            # START EPISODE
            # =================

            # Initialize: first action
            observation = env.reset()

            # Declare many lists
            prob_list = []
            reward_list = []
            action_list = []
            # Store last observation so that we can compute delta_obs
            last_observation = observation
            action = env.get_random_action()
            # ==================
            # PLAY UNTIL DIE
            # ==================

            done = False
            total_round = 0
            while done != True:
                observation, reward, done, info = env.step(self.action_dict[action]) # map 0 to UP(2), map 1 to DOWN(5)
                delta_observation = prepro(observation) - prepro(last_observation)
                delta_observation = torch.from_numpy(delta_observation).type(torch.float).unsqueeze(0).to(self.device)
                last_observation = observation
                # prediction
                down_prob = model(delta_observation)
                # sample an action
                action = 1 if np.random.uniform() < down_prob.item() else 0
                # push data into list
                prob_list.append(down_prob)
                reward_list.append(reward)
                action_list.append(action)
                total_round += 1
            total_rewards = sum(reward_list)

            self.optimizer.zero_grad()
            # discount factor
            discount_reward = self._discount_reward(reward_list)
            probs = torch.cat((prob_list), dim = 0).to(self.device)
            actions, rewards = torch.tensor(action_list).type(torch.float).to(self.device), torch.tensor(discount_reward).type(torch.float).to(self.device)

            loss = self.criterion(probs.squeeze(1), actions, weight = rewards)
            loss.backward()
            self.optimizer.step()
            
            self.logger.info('episode: %d total_rewards: %.2f total_round: %d loss: %f' % (episode_n, total_rewards, total_round, loss.item()))

            log = {
                'episode': episode_n,
                'loss': loss.item(),
                'total_round': total_round,
                'total_rewards': total_rewards
            }

            self.log_list.append(log)
 
            if episode_n % (save_n ** 2) == 0:
                save_n += 1
                checkpoint = {
                    'log': self.log_list,
                    'state_dict': self.model.state_dict()
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint_episode{}.pth.tar'.format(episode_n)))
                self.logger.info('save checkpoint_episode{}.pth.tar!'.format(episode_n))

                elapsed_time = time.time() - self.start_time
                self.logger.info('consume {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            episode_n += 1

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.check_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.log_list = checkpoint['log']
        self.start_episode = checkpoint['log'][-1]['episode'] + 1
    def _discount_reward(self, r):
        dr = np.zeros_like(r)
        curr = 0
        for i in reversed(range(len(r))):
            if r[i] != 0:
                curr = r[i]
            else:
                curr = curr * 0.99
            dr[i] = curr
        dr = (dr - np.mean(dr)) / np.std(dr)
        return dr
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        delta_observation = prepro(observation) - prepro(self.test_last_observation) if self.test_last_observation is not None else None
        self.test_last_observation = observation
        if delta_observation is None:
            return 0
        # prediction
        delta_observation = torch.from_numpy(delta_observation).type(torch.float).unsqueeze(0).to(self.device)
        down_prob = self.model(delta_observation)
        # sample an action
        action = 1 if np.random.uniform() < down_prob.item() else 0
        return self.action_dict[action]


