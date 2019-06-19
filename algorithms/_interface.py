import torch
import os

class RLInterface:
    def __init__(self):
        self.global_network = None
        self.env_name = None
        self.save_load_path = "trained_models"

    def run(self):
        raise NotImplementedError()
        return

    def save(self):
        save_path = os.path.join(self.save_load_path, self.env_name + '.pth')
        torch.save(self.global_network.state_dict(), save_path)
        print("Model saved to %s." % save_path)

    def load(self):
        load_path = os.path.join(self.save_load_path, self.env_name + '.pth')
        if os.path.isfile(load_path):
            self.global_network.load_state_dict(torch.load(load_path, map_location='cpu'))
            print("Model loaded from %s." % load_path)