import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from configs import Config


class Trainer:
    def __init__(self, model, optimizer, checkpointer, train_dataloader, test_dataloader, tester, logs_path):
        self.work_threads = Config.n_threads
        self.train_times = Config.n_epochs

        self.opt_method = Config.opt_method
        self.lr_decay = Config.lr_decay
        self.weight_decay = Config.weight_decay
        self.alpha = Config.alpha

        self.use_gpu = Config.use_gpu
        self.save_steps = Config.save_epochs
        self.test_steps = Config.test_epochs

        self.model = model
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.tester = tester

        self.benchmark = Config.benchmark
        self.tb_writer = SummaryWriter(log_dir=logs_path)

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        training_range = tqdm(range(1, self.train_times + 1))
        for epoch in training_range:
            res = 0.0
            for data in self.train_dataloader:
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description('Epoch %d | loss: %f' % (epoch, res))

            self.tb_writer.add_scalar(f'{self.benchmark}/loss', res, epoch)

            if epoch % self.test_steps == 0:
                tester = self.tester(model=self.model.model, data_loader=self.test_dataloader, use_gpu=self.use_gpu)
                metrics = tester.run_link_prediction(type_constrain=Config.type_constrain)

                for cons in metrics:
                    for metric in metrics[cons]:
                        self.tb_writer.add_scalar(f'{self.benchmark}/{cons}/{metric}', metrics[cons][metric], epoch)

            if self.checkpointer is not None and epoch % self.save_steps == 0:
                self.checkpointer.save(self.model.model, self.optimizer, epoch)

    @staticmethod
    def to_var(x, use_gpu):
        if use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)
