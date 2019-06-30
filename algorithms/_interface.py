import torch
import os
import logging
from jsonmanager import JsonManager
from tensorboardX import SummaryWriter
import atexit


class RLInterface:
    def __init__(self):
        self.name = "UndefinedAlgorithm"
        self.logprefix = ""
        self.global_network = None
        self.env_name = None
        self.save_load_path = "trained_models"
        self.save_path = True
        self.jsonmanager = None
        self.last_max_reward = float("-inf")
        self.isTraining = True
        self.env_factory = None

        self.writer = None
        atexit.register(self.on_exit)

    def run(self):
        raise NotImplementedError()
        return

    def play(self):
        raise NotImplementedError()
        return

    def save(self):
        if not self.save_path:
            return

        save_path = os.path.join(self.save_load_path, self.name + '_' + self.env_name + '.pth')
        torch.save(self.global_network.state_dict(), save_path)
        self.jsonmanager.set("last_max_reward", self.last_max_reward)        
        logging.info(self.logprefix + "Model saved to %s." % save_path)

    def load(self):
        load_path = os.path.join(self.save_load_path, self.name + '_' + self.env_name + '.pth')

        if os.path.isfile(load_path):
            self.global_network.load_state_dict(torch.load(load_path, map_location='cpu'))
            logging.info(self.logprefix + "Model loaded from %s." % load_path)

        last = self.jsonmanager.get("last_max_reward")
        self.last_max_reward =  last if last is not None else float("-inf")
    
    def init_jsonmanager(self):
        path = os.path.join(self.save_load_path, self.name + '_' + self.env_name + '.json')
        self.jsonmanager = JsonManager(path)

    def init_writer(self):
        self.writer = SummaryWriter(comment="-" + self.name + "_" + self.env_name)

    def record(self, scalar, argX, argY, message=""):
        if self.isTraining:
            logging.info(self.logprefix + "Episode %d  |  %s -  Reward: %.0f" % (argX, message, argY))
            
            if self.last_max_reward < argY:
                self.last_max_reward = argY
                self.save()
        else:
            logging.info(self.logprefix + "Game %d | %s - Reward: %.0f" % (argX, message, argY))

        if self.writer is None:
            logging.error(self.logprefix + "Tensorboard Writter was not initialized.")
            return
        else: 
            self.writer.add_scalar(scalar, argY, argX)  


    def on_exit(self):
        if self.writer is not None:
            logging.info(self.logprefix + "Saving tensorboard...")
            self.writer.close()

        logging.info(self.logprefix + "Exiting")