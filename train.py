import os
import shutil
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Select an experiment name and override values in its config')

    parser.add_argument('--exp', type=str, help='Name of experiment', required=True)
    parser.add_argument('--benchmark', type=str, help='Name of benchmark', required=False)
    parser.add_argument('--n_epochs', type=int, help='Number of epochs', required=False)

    return parser


def prepare_run(args):
    shutil.copyfile(os.path.join('configs', 'experiments', args.exp + '.py'), os.path.join('configs', '__init__.py'))

    from configs import Config
    for key, value in vars(args).items():
        if value is not None:
            setattr(Config, key, value)
            if key == 'benchmark':
                setattr(Config, 'data_path', f'./benchmarks/{value}/')


def run():
    parser = create_parser()
    args = parser.parse_args()
    prepare_run(args)

    from configs import Components
    from builder import Builder

    builder = Builder()
    builder.build()
    trainer = Components.trainer(**builder.get_trainer_arguments())
    del builder
    trainer.run()


if __name__ == '__main__':
    run()
