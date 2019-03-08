import os
import json
import argparse
import torch
import sys
sys.path.append('../../')
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import HessianTrainer
from utils import Logger


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config):
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
    trainer = HessianTrainer(model, loss, metrics, optimizer, 
                      resume=False,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)

    trainer.train()

def hardcode_config():
    config = {
        "name": "Hessian",
        "n_gpu": 1,
        
        "arch": {
            "type": "DNN8",
            "args": {}
        },
        "data_loader": {
            "type": "CsvDataLoader",
            "args":{
                "data_dir": "data/",
                "batch_size": 512,
                "shuffle": True,
                "validation_split": 0,
                "num_workers": 2
            }
        },
        "optimizer": {
            "type": "Adam",
            "args":{
                "lr": 0.001,
                "weight_decay": 0,
                "amsgrad": True
            }
        },
        "loss": "L2_loss",
        "metrics": [],
        "lr_scheduler": {
            "type": "StepLR",
            "args": {
                "step_size": 50,
                "gamma": 0.1
            }
        },
        "trainer": {
            "epochs": 200,
            "save_dir": "saved/",
            "save_period": 200,
            "verbosity": 2,
            
            "monitor": "min val_loss",
            "early_stop": 1e10,
            
            "tensorboardX": False,
            "log_dir": "saved/runs"
        }
    }
    return config

if __name__ == '__main__':
    config = hardcode_config()
    for i in range(9, 100):
        config['name'] = str(i)
        main(config)
