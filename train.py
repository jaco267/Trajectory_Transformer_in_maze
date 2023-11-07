import torch
from z_train.training import Trainer
from traject.data_preprocess.sequence import SequenceDataset
from traject.models.transformers import GPT
import pyrallis    #nice cmd config tool
from utils.save import savedata_config,load_data_config
from config_train import TrainConfig, GPT_config,Trainer_config  ##*must include these
from config_data import DataConfig
import os
from traject.utils.serialization import mkdir
@pyrallis.wrap()    
def main(targs: TrainConfig):
  dargs:DataConfig = load_data_config(targs.data_path,'data_config.pkl')
  if mkdir(targs.savepath): print(f'[ utils/setup ] Made savepath: {targs.savepath}')
  print('####### dataset #######')
  dataset = SequenceDataset(dargs=dargs,targs=targs)
  print(f'Dataset size: {len(dataset)}, \
dim {dataset.observation_dim, dataset.action_dim,dataset.joined_dim}') 
  print('######## model ########')  
  ##                      obs=w_h**2 , action = 1, joined=w_h**2+1+1+1 = 19 (obs action,reward,done)
  targs.update_config(len(dataset),dataset.observation_dim,dataset.action_dim,dataset.joined_dim)
  savedata_config(savepath=(targs.savepath,'train_config.pkl'),args=targs)
  model = GPT(targs.gpt_config,dargs).to(targs.device);
  print('####### trainer #######')
  #tokens per epoch=len(dataset)* (seq_len*state_dim)
  
  # breakpoint()
  trainer = Trainer(targs.trainer_config,dargs)
  ###### main loop ######
  ## scale number of epochs to keep number of updates constant
  n_epochs =  targs.n_epochs_ref     
  save_freq = int(n_epochs // targs.n_saves)
  for epoch in range(n_epochs):  #50
      print(f'\nEpoch: {epoch} / {n_epochs} ')
      trainer.train(model, dataset)
      ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
      save_epoch = (epoch + 1) // save_freq * save_freq
      statepath = os.path.join(targs.savepath, f'state_{save_epoch}.pt')
      print(f'Saving model to {statepath}')
      torch.save(model.state_dict(), statepath)
if __name__ == '__main__':
    main()