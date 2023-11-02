import os
import cloudpickle as pickle   #we use cloudpickle because pickle cant store dataclase properly
from config_train import TrainConfig,Trainer_config,GPT_config
def savedata_config(savepath,args):
    savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
    pickle.dump(args, open(savepath, 'wb'))
    print(f'Saved config to: {savepath}\n')

def load_data_config(*loadpath):
    loadpath = os.path.join(*loadpath) 
    #'logs/maze/gpt/pretrained/train_config.pkl'
    config:TrainConfig = pickle.load(open(loadpath, 'rb'))
    print(f'Loaded data config from {loadpath}')
    print(config)
    return config