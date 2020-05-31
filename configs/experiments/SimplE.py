class Config:
    # DATA
    benchmark = 'FB15K237'
    data_path = f'./benchmarks/{benchmark}/'

    # DEVICE
    use_gpu = True

    # MODEL
    embedding_dim = 200

    # TRAIN DATALOADER
    tri_file = None
    ent_file = None
    rel_file = None
    batch_size = None
    n_batches = 100
    n_threads = 8
    train_sampling_mode = 'normal'
    bern_flag = 1
    filter_flag = 1
    neg_ent = 25
    neg_rel = 0

    # TRAINING
    n_epochs = 1000
    alpha = 0.5
    opt_method = 'adagrad'
    weight_decay = 0
    lr_decay = 0
    regul_rate = 1.
    l3_regul_rate = 0.

    # TEST DATALOADER
    test_sampling_mode = 'link'
    test_epochs = 50
    type_constrain = True

    # LOGS & CHECKPOINTS
    save_epochs = 100
    logs_path = 'logs/'
    logs_file = 'run.log'
    checkpoints_folder = 'checkpoints/'
    date_format = '%y-%m-%d'
    time_format = '%H-%M-%S'
    log_format = '%(levelname)s - %(module)s:%(funcName)s:%(lineno)s - %(asctime)s - %(message)s'
    log_time_format = date_format + ':' + time_format


class Components:
    from openke.module.model import SimplE
    from openke.module.loss import SoftplusLoss
    from openke.data import TrainDataLoader, TestDataLoader
    from openke.module.strategy import NegativeSampling
    from openke.config import Trainer, Tester

    train_dataloader = TrainDataLoader
    test_dataloader = TestDataLoader
    model = SimplE
    loss = SoftplusLoss()
    strategy = NegativeSampling
    trainer = Trainer
    tester = Tester
