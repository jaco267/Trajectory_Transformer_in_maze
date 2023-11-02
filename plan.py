import json; 
from os.path import join

import traject.utils as utils
from traject.data_preprocess.sequence import SequenceDataset
from traject.search import ( beam_plan, make_prefix, extract_actions, update_context, )
from traject.models.transformers import GPT
import pyrallis
import math
import torch

from config_plan import PlanConfig
from maze.env_maze.env import ProcMaze
import numpy as np
from z_plan.utils import load_model
from z_plan.main import main_loop
from utils.save import load_data_config
def maze_state_to_obs(maze_state):
  goal, wall_grid, pos, _ = maze_state
  wall_grid = wall_grid.to(torch.float32)
  wall_grid[goal[0],goal[1]] += 4
  wall_grid[pos[0],pos[1]] += 2
  obss = wall_grid.reshape(-1).numpy()
  return obss



@pyrallis.wrap()    
def main(args: PlanConfig):  
  gpt_folder = args.gpt_folder
  targs = load_data_config(gpt_folder, 'train_config.pkl')
  if args.seed != None:  torch.random.manual_seed (args.seed)
  #train_config
  sequence_length = targs.subsampled_sequence_length * targs.step  #10*1 # same as script/train.py
  dataset = SequenceDataset(data_path=targs.data_path,N=targs.N,
            max_path_length=targs.max_path_length,
            sequence_length=sequence_length,
            step=targs.step,discount=targs.discount)
  config, ckpt, gpt_epoch = load_model(gpt_folder, 
            config_file='train_config.pkl', epoch=args.gpt_epoch)  #todo  merge two train config
  config.update_config(len(dataset),# fix bug of pyrallis (reset the gpt config)
                dataset.observation_dim, dataset.action_dim, dataset.joined_dim)
  # breakpoint()#config.gpt_config.block_size #config.observation_dim=7 #config.n_layer
  config = config.gpt_config
  model = GPT(config)
  model.to(args.device)
  model.load_state_dict(ckpt, strict=True)
  ####### dataset #######
  # model.observation_dim = 16# model.transition_dim = 19 = 16+1+1+1
  observation_dim = dataset.observation_dim;  #** w*h  
  w_h = int(observation_dim**0.5)
  timeout=50
  env_maze = ProcMaze(grid_size=w_h,device='cpu',timeout=timeout)

  score,t,total_reward,terminal = main_loop(args,env_maze,timeout,dataset,gpt_model=model) 
  ## save result as a json file
  json_path = join(args.savepath, 'rollout.json')
  json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch}
  json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)



if __name__ == '__main__':
   main()