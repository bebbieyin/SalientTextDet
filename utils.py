import yaml
import os
import torch 
import pandas as pd
import argparse
import csv


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type = str,required = True)
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    return config

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config


def create_logs(fieldnames, filepath):

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def save_results(csv_path, results, fieldnames):

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(results)

def save_checkpoint(config, epoch, model, optimizer, lr_scheduler):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config['OUTPUT']['OUT_PATH'],'model_{}.pt'.format(epoch))

    torch.save(save_state, save_path)

def load_checkpoint(ckpt_path, model, optimizer, scheduler):

    checkpoint = torch.load(ckpt_path)
    weights = checkpoint['model']
    model.load_state_dict(weights)

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])

    return model, optimizer, scheduler, checkpoint['epoch']

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
     
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
