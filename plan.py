import sys; import os
cwd = os.getcwd(); sys.path.append(cwd)
import json; from os.path import join

import traject.utils as utils
from traject.data_preprocess.sequence import SequenceDataset
from traject.search import ( beam_plan, make_prefix, extract_actions, update_context, )
from traject.models.transformers import GPT
import pyrallis
import glob
import math
import torch
import cloudpickle as pickle  #we use cloudpickle because pickle cant store dataclase properly
from config_train import TrainConfig,Trainer_config,GPT_config
#import Trainer_config is necessary because it is the dataclass inside Trainconfig
from config_plan import PlanConfig
from maze.env_maze.env import ProcMaze
import numpy as np
#0: no_op, 1: up, 2: left, 3: down, 4: right
action_map = {
   0:'no_op',
   1:'up',
   2:'left',
   3:'down',
   4:'right'
}
def load_data_config(*loadpath):
    loadpath = os.path.join(*loadpath) 
    #'logs/maze/gpt/pretrained/train_config.pkl'
    config:TrainConfig = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def get_latest_epoch(loadpath):
    states = glob.glob1(loadpath, 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch
def load_model(*loadpath,model_file='model_config.pkl', epoch=None):
    loadpath = os.path.join(*loadpath)
    config_path = os.path.join(loadpath, model_file)

    if epoch is 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'[ utils/serialization ] Loading model epoch: {epoch}')
    state_path = os.path.join(loadpath, f'state_{epoch}.pt')

    config:GPT_config = pickle.load(open(config_path, 'rb'))
    state = torch.load(state_path)
    print(f'\n[ utils/serialization ] Loaded config from {config_path}\n')
    return config, state,  epoch
def maze_state_to_obs(maze_state):
  goal, wall_grid, pos, _ = maze_state
  wall_grid = wall_grid.to(torch.float32)
  wall_grid[goal[0],goal[1]] += 4
  wall_grid[pos[0],pos[1]] += 2
  obss = wall_grid.reshape(-1).numpy()
  return obss
@pyrallis.wrap()    
def main(args: PlanConfig):  
  ####### models ########  rl envs offline_data
  gpt_folder = 'logs/maze/gpt/pretrained'
  targs = load_data_config(gpt_folder, 'train_config.pkl')
  if args.seed != None:  torch.random.manual_seed (args.seed)
  #train_config
  sequence_length = targs.subsampled_sequence_length * targs.step  #10*1 # same as script/train.py
  dataset = SequenceDataset(data_path=targs.data_path,N=targs.N,
            penalty=targs.termination_penalty,
            max_path_length=targs.max_path_length,
            sequence_length=sequence_length,
            step=targs.step,discount=targs.discount)
  config, state, gpt_epoch = load_model(gpt_folder, 
            model_file='train_config.pkl', epoch=args.gpt_epoch)  #todo  merge two train config
  _ = config.update_config(len(dataset),# fix bug of pyrallis (reset the gpt config)
                dataset.observation_dim, dataset.action_dim, dataset.joined_dim)
  # breakpoint()#config.gpt_config.block_size #config.observation_dim=7 #config.n_layer
  config = config.gpt_config
  model = GPT(config)#config()
  model.to(args.device)
  model.load_state_dict(state, strict=True)
  print(config)
  gpt = model
  ####### dataset #######
  # gpt.observation_dim = 16# gpt.transition_dim = 19 = 16+1+1+1
  discretizer = dataset.discretizer
  discount = dataset.discount
  observation_dim = dataset.observation_dim;  action_dim = dataset.action_dim
  timeout=50
  env_maze = ProcMaze(grid_size=int(observation_dim**0.5),device='cpu',timeout=timeout)
  timer = utils.timer.Timer()
  
  
  
  value_fn = lambda x: discretizer.value_expectation(x)
  #######################
  ###### main loop ######
  #######################
  maze_state = env_maze.reset();
  obss = maze_state_to_obs(maze_state)
  #***
  total_reward = 0
  rollout = [obss.copy()]## observations for rendering
  context = []## previous (tokenized) transitions for conditioning transformer
  T = timeout  #*1000
  for t in range(T):
    ## concatenate previous transitions and current observations to input to model
    prefix = make_prefix(discretizer, context, obss, args.prefix_context)
    # # [1,26~196]                    []      (26,)          True
    ## sample sequence_ from model beginning with `prefix`
    sequence = beam_plan(
        gpt, value_fn, prefix,
        args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
        discount, args.max_context_transitions, verbose=args.verbose,
        k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
    )
    ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
    sequence_recon = discretizer.reconstruct(sequence)

    ## [ action_dim ] index into sampled trajectory to grab first action
    action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)

    ## execute action in environment
    # next_observation, reward, terminal, _ = env.step(action)
    print(action,"should be 0~4")
    action_in = np.clip(math.floor(action.item()), 0, 5 - 1)  #action dim
    maze_state, _, reward, terminal, _ = env_maze.step(action_in, maze_state)
    next_obss = maze_state_to_obs(maze_state)
    print(obss.reshape(4,4),action_map[action_in])
  
    ## update return
    total_reward += reward
    score = total_reward

    ## update rollout observations and context transitions
    rollout.append(next_obss.copy())
    context = update_context(context, discretizer, obss, action, reward, 
                             args.max_context_transitions,value=sequence_recon[0][-1])
    #todo prefix context's value should be generated by gpt
    if args.enable_breakpoint: 
      breakpoint()
    print(
        f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
        f'time: {timer():.2f}  | {args.exp_name} | {args.suffix}\n'
    )
    if terminal: break
    obss = next_obss

  ## save result as a json file
  json_path = join(args.savepath, 'rollout.json')
  json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal.item(), 'gpt_epoch': gpt_epoch}
  json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)



if __name__ == '__main__':
   main()