import torch
import os
import logging
from tensorboardX import SummaryWriter
import atexit
import time
import random
import numpy as np
from utils import np_torch_wrap
import sys


class RLInterface:
    def __init__(self):
        self.name = "UndefinedAlgorithm"
        self.logprefix = ""
        self.global_network = None
        self.optimizer = None
        self.env_name = None
        self.episode = 0  # needed to save/load and resume training
        self.save_load_path = "trained_models"
        self.time_start_run = 0
        self.last_max_reward = float("-inf")
        self.is_training = True
        self.env_factory = None
        self.writer = None
        atexit.register(self.on_exit)

    def run(self):
        self.time_start_run = time.time()

    def play(self, game_plays):
        self.is_training = False
        
        logging.info(self.logprefix + 'Playing game...')

        reward_mean = []
        for i in range(game_plays):
            env = self.env_factory[random.randint(0, len(self.env_factory) - 1)]() if type(self.env_factory) is list else self.env_factory()

            state = env.reset()
            terminal = False
            game_reward = 0
            while not terminal:
                if self.render:
                    env.render()
                    time.sleep(0.03)

                action_index = self.global_network.choose_action(np_torch_wrap(state[None, :]))
                state, reward, terminal, info = env.step(action_index)
                game_reward += reward
            
            self.record(
                message=self.env_name,
                episode=i+1,
                reward=game_reward
            )
            reward_mean.append(game_reward)
            env.close()
        
        mean = np.array(reward_mean).mean()
        logging.info(self.logprefix + 'Reward mean ' + str(mean))

    def save(self):
        if not self.is_training:
            return

        save_path = os.path.join(self.save_load_path, self.name + '_' + self.env_name + '.pth')

        # backup last saved model in case we corrupt this one
        sys.exec('mv ' + save_path + ' ' + save_path + '.old')

        state = {
            'network': self.global_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'max_reward': self.last_max_reward,
            'episode': self.episode
        }
        torch.save(state, save_path)

        logging.info(self.logprefix + "Model %d saved to %s." % (self.episode, save_path))

    def load(self):
        load_path = os.path.join(self.save_load_path, self.name + '_' + self.env_name + '.pth')

        if os.path.isfile(load_path):
            state = torch.load(load_path, map_location='cpu')
            self.global_network.load_state_dict(state['network'])
            self.global_network.train()
            self.global_network.eval()
            self.optimizer.load_state_dict(state['optimizer'])
            self.last_max_reward = state['max_reward']
            self.episode = state['episode']

            logging.info(self.logprefix + "Model loaded from %s." % load_path)

    def init_writer(self):
        self.writer = SummaryWriter(comment="-" + self.name + "_" + self.env_name)

    def record(
        self, 
        message, 
        episode, 
        reward, 
        episode_length=False, 
        mean_loss=False, 
        mean_value_loss=False,
        mean_policy_loss=False,
        mean_advantage=False,
        # mean_predicted_value=False,
        gradient_updates=False):

        if self.writer is None:
            logging.error(self.logprefix + "Tensorboard Writter was not initialized.")
            return

        time_elapsed = time.time() - self.time_start_run

        logging.info(
            self.logprefix + 
            "Episode: " + str(episode) + "  |  " +
            "Time elapsed: " + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)) + "  |  " +
            message + "  |  " +
            "Reward: " + "{0:.2f}".format(reward) +
            ("  |  Mean Loss:" + "{0:.2f}".format(mean_loss) if mean_loss else "")
        )

        if self.is_training:
            if self.last_max_reward == float('-inf') or int(self.last_max_reward) < int(reward):
                self.last_max_reward = reward
                self.save()

            self.writer.add_scalar("Time/Episode", time_elapsed, episode)
            if reward: 
                self.writer.add_scalar("Reward/Episode", reward, episode)
            if mean_loss:
                self.writer.add_scalar("MeanLoss/Episode", mean_loss, episode)
            if gradient_updates:
                self.writer.add_scalar("GradientUpdates/Episode", gradient_updates, episode)
            if episode_length:
                self.writer.add_scalar('EpisodeLength/Episode', episode_length, episode)
            if mean_value_loss:
                self.writer.add_scalar('MeanValueLoss/Episode', mean_value_loss, episode)
            if mean_policy_loss:
                self.writer.add_scalar('MeanPolicyLoss/Episode', mean_policy_loss, episode)
            if mean_advantage:
                self.writer.add_scalar('MeanAdvantage/Episode', mean_advantage, episode)
            #if mean_predicted_value:
            #    self.writer.add_scalar('Mean Predicted Value / Episode', mean_predicted_value, episode)
        else: 
            self.writer.add_scalar("TestGameReward", reward, episode)

    def on_exit(self):
        if self.writer is not None:
            logging.info(self.logprefix + "Saving tensorboard...")
            self.writer.close()

        logging.info(self.logprefix + "Exiting")