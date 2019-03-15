import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger

def handle_args(config, args):
    # use argument to overwrite the original config setting
    if args.type != None:
        print(args.type)
        config['arch']['type'] = args.type
        config['name'] = args.type
    if args.name != None:
        config['name'] = args.name
    if args.save != None:
        config['trainer']['save_dir'] = args.save
    if args.epoch != None:
        config['trainer']['epochs'] = int(args.epoch)
    if args.period != None:
        config['trainer']['save_period'] = args.period
    if args.shuffle_ratio != None:
        config['data_loader']['args']['shuffle_ratio'] = args.shuffle_ratio
    if args.c1 != None:
        config['arch']['args']['c1'] = args.c1
    if args.c2 != None:
        config['arch']['args']['c2'] = args.c2
    if args.c3 != None:
        config['arch']['args']['c3'] = args.c3
    if args.c4 != None:
        config['arch']['args']['c4'] = args.c4

    return config

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    print(model)
    
    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    trainer = Trainer(model, loss, metrics, optimizer, 
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=None,
                      train_logger=train_logger)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--type', default=None, type=str,
                           help='model type')
    parser.add_argument('--name', default=None, type=str,
                           help='model name')
    parser.add_argument('--save', default=None, type=str,
                           help='save dir')
    parser.add_argument('--epoch', default=None, type=str,
                           help='epoch number')
    parser.add_argument('--period', default=None, type=int, help='save_period')
    parser.add_argument('--shuffle_ratio', default=None, type=float,
                           help='shuffle_ratio')
    parser.add_argument('--c1', default=None, type=float, help='c1')
    parser.add_argument('--c2', default=None, type=float, help='c2')
    parser.add_argument('--c3', default=None, type=float, help='c3')
    parser.add_argument('--c4', default=None, type=float, help='c4')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        config = handle_args(config, args)
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
