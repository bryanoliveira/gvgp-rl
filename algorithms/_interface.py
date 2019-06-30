import torch
import os
import logging
from tensorboardX import SummaryWriter
import atexit
import time


class RLInterface:
    def __init__(self):
        self.name = "UndefinedAlgorithm"
        self.logprefix = ""
        self.global_network = None
        self.env_name = None
        self.save_load_path = "trained_models"
        self.time_start_run = 0

        self.writer = None
        atexit.register(self.on_exit)

    def run(self):
        self.time_start_run = time.time()

    def save(self):
        save_path = os.path.join(self.save_load_path, self.name + '_' + self.env_name + '.pth')
        torch.save(self.global_network.state_dict(), save_path)
        logging.info(self.logprefix + "Model saved to %s." % save_path)

    def load(self):
        load_path = os.path.join(self.save_load_path, self.name + '_' + self.env_name + '.pth')
        if os.path.isfile(load_path):
            self.global_network.load_state_dict(torch.load(load_path, map_location='cpu'))
            logging.info(self.logprefix + "Model loaded from %s." % load_path)

    def init_writer(self):
        self.writer = SummaryWriter(comment="-" + self.name + "_" + self.env_name)

    def record(
        self, 
        message, 
        episode, 
        reward=False, 
        episode_length=False, 
        mean_loss=False, 
        mean_value_loss=False,
        mean_policy_loss=False,
        mean_advantage=False,
        # mean_predicted_value=False,
        gradient_updates=False):

        time_elapsed = time.time() - self.time_start_run

        logging.info(
            self.logprefix + 
            "Episode: " + str(episode) + "  |  " +
            "Time elapsed: " + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)) + "  |  " +
            message + "  |  " +
            ("Reward: " + "{0:.2f}".format(reward) + "  |  " if reward else "") +
            ("Mean Loss:" "{0:.2f}".format(mean_loss) if mean_loss else "")
        )

        if self.writer is None:
            logging.error(self.logprefix + "Tensorboard Writter was not initialized.")
            return
        else:
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

    def on_exit(self):
        self.save()  # save last state of our network

        if self.writer is not None:
            logging.info(self.logprefix + "Saving tensorboard...")
            self.writer.close()

        logging.info(self.logprefix + "Exiting")