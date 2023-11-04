import json; 
from os.path import join
from traject.data_preprocess.sequence import SequenceDataset
from traject.models.transformers import GPT
import pyrallis
import torch
from config_data import DataConfig
from config_plan import PlanConfig
from maze.env_maze.env import ProcMaze
from z_plan.utils import load_model
from utils.save import load_data_config
from z_plan.main import main_loop


def maze_state_to_obs(maze_state):
  goal, wall_grid, pos, _ = maze_state
  wall_grid = wall_grid.to(torch.float32)
  wall_grid[goal[0],goal[1]] += 4
  wall_grid[pos[0],pos[1]] += 2
  obss = wall_grid.reshape(-1).numpy()
  return obss



@pyrallis.wrap()    
def main(args: PlanConfig):  
  dargs:DataConfig = load_data_config(args.data_path,'data_config.pkl')
  gpt_folder = args.gpt_folder
  targs = load_data_config(gpt_folder, 'train_config.pkl')
  if args.seed != None:  torch.random.manual_seed (args.seed)
  #train_config
  dataset = SequenceDataset(dargs=dargs,targs=targs)
  # breakpoint()
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
  # model.observation_dim = w_h**2# model.transition_dim = w_h**2+1+1+1
  observation_dim = dataset.observation_dim;  #** w*h  
  w_h = int(observation_dim**0.5)
  timeout=args.timeout
  env_maze = ProcMaze(grid_size=w_h,device='cpu',timeout=timeout)

  t,total_reward,terminal = main_loop(dargs,args,env_maze,timeout,dataset,gpt_model=model) 
  ## save result as a json file
  json_path = join(args.savepath, 'rollout.json')
  json_data = {'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch}
  json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)



if __name__ == '__main__':
   main()