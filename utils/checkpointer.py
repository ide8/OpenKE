import os

import torch
from transformers import PreTrainedModel
from torch.optim.optimizer import Optimizer


class Checkpointer:
    def __init__(self, path):
        if not os.path.exists(path):
            raise ValueError(f'Checkpointer root path f{path} does not exist')
        self.root = path
        self.active = True

    def epoch_step_path(self, epoch, step=None):
        return os.path.join(self.root, f'epoch_{epoch}{(f"_{step}" if step else "")}')

    def model_checkpoint_fn(self, epoch, step=None):
        return os.path.join(self.epoch_step_path(epoch, step), 'model.pt')

    def optimizer_checkpoint_fn(self, epoch, step=None):
        return os.path.join(self.epoch_step_path(epoch, step), 'optimizer.pt')

    def save(self, model, optimizer, epoch: int, step: int = None):
        if not self.active:
            return
        os.makedirs(self.epoch_step_path(epoch, step))
        torch.save(model.state_dict(), self.model_checkpoint_fn(epoch, step))
        torch.save(optimizer.state_dict(), self.optimizer_checkpoint_fn(epoch, step))

    def load(self, epoch, step=None):
        model_state_dict = torch.load(self.model_checkpoint_fn(epoch, step), map_location='cpu')
        optimizer_state_dict = torch.load(self.optimizer_checkpoint_fn(epoch, step), map_location='cpu')

        return model_state_dict, optimizer_state_dict
