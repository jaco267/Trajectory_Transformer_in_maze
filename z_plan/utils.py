import os
import glob
import torch
import cloudpickle as pickle  #we use cloudpickle because pickle cant store dataclase properly
from config_train import TrainConfig,Trainer_config,GPT_config
#import Trainer_config is necessary because it is the dataclass inside Trainconfig??

def load_model(*loadpath,config_file, epoch=None):
    def get_latest_epoch(loadpath):
        states = glob.glob1(loadpath, 'state_*')
        latest_epoch = -1
        for state in states:
            epoch = int(state.replace('state_', '').replace('.pt', ''))
            latest_epoch = max(epoch, latest_epoch)
        return latest_epoch
    loadpath = os.path.join(*loadpath)
    if epoch is 'latest': epoch:int = get_latest_epoch(loadpath)
    print(f'Loading model state epoch: {epoch}')
    state_path = os.path.join(loadpath, f'state_{epoch}.pt')
    ckpt = torch.load(state_path)

    config_path = os.path.join(loadpath, config_file)
    print(f'\nLoaded train config from {config_path}\n')
    config:GPT_config = pickle.load(open(config_path, 'rb'))
    return config, ckpt,  epoch