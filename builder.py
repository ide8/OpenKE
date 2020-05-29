import logging
import os
import time
from shutil import copyfile

import torch.optim as optim

from utils.checkpointer import Checkpointer
from configs import Config, Components


class Builder:
    def __init__(self):
        self.base_logs_path = os.path.join(Config.logs_path, Config.exp)

        self.logs_path = None
        self.logs_file_path = None
        self.log_handlers = []

        self.checkpointer = None

        self.model = None
        self.optimizer = None

        self.train_dataloader = None
        self.test_dataloader = None

        self.tester = Components.tester

    def build(self):
        self.init_log_folder()
        self.add_file_log_handler()
        self.add_stream_log_handler()
        self.create_logger()
        self.create_checkpointer()
        self.create_dataloaders()
        self.create_model()
        self.create_otimizer()

    def init_log_folder(self):
        self.logs_path = os.path.join(
            self.base_logs_path, time.strftime(Config.date_format), time.strftime(Config.time_format))
        os.makedirs(self.logs_path)

        copyfile(os.path.join('configs', 'experiments', Config.exp + '.py'), os.path.join(self.logs_path, 'configs.py'))

    def add_file_log_handler(self):
        self.logs_file_path = os.path.join(self.logs_path, Config.logs_file)
        self.log_handlers.append(logging.FileHandler(self.logs_file_path))

    def add_stream_log_handler(self):
        self.log_handlers.append(logging.StreamHandler())

    def create_logger(self):
        logging.basicConfig(level=logging.INFO, format=Config.log_format,
                            datefmt=Config.log_time_format, handlers=self.log_handlers)
        logging.info(f'Log file initialized at {self.logs_file_path}')

    def create_checkpointer(self):
        checkpoints_path = os.path.join(self.logs_path, Config.checkpoints_folder)
        os.makedirs(checkpoints_path)

        self.checkpointer = Checkpointer(checkpoints_path)
        logging.info(f'Checkpointer initialized at {checkpoints_path}')

    def create_dataloaders(self):
        self.train_dataloader = Components.train_dataloader()
        self.test_dataloader = Components.test_dataloader()

    def create_model(self):
        self.model = Components.model(
            ent_tot=self.train_dataloader.get_ent_tot(),
            rel_tot=self.test_dataloader.get_rel_tot(),
        )

        self.model = Components.strategy(
            model=self.model,
            loss=Components.loss,
            batch_size=self.train_dataloader.get_batch_size()
        )

        if Config.use_gpu:
            self.model.cuda()

    def create_otimizer(self):
        if Config.opt_method.lower() == 'adagrad':
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=Config.alpha,
                lr_decay=Config.lr_decay,
                weight_decay=Config.weight_decay,
            )
        elif Config.opt_method.lower() == 'adadelta':
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=Config.alpha,
                weight_decay=Config.weight_decay,
            )
        elif Config.opt_method.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=Config.alpha,
                weight_decay=Config.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=Config.alpha,
                weight_decay=Config.weight_decay,
            )

    def get_trainer_arguments(self):
        return {
            key: getattr(self, key)
            for key in (
                'model', 'optimizer', 'checkpointer', 'train_dataloader', 'test_dataloader', 'tester', 'logs_path'
            )
        }
