from traject.utils.timer import Timer

from z_plan.search import ( beam_plan, make_prefix, extract_actions, update_context, )
import math
import torch
import numpy as np
from config_data import DataConfig
def maze_state_to_obs(maze_state):
  goal, wall_grid, pos, _ = maze_state
  wall_grid = wall_grid.to(torch.float32)
  wall_grid[goal[0],goal[1]] += 4
  wall_grid[pos[0],pos[1]] += 2
  obss = wall_grid.reshape(-1).numpy()
  return obss
def main_loop(dargs,args,env_maze,timeout,dataset,gpt_model):
  action_map:dict = env_maze.action_map_dict #0: no_op, 1: up, 2: left, 3: down, 4: right
  discretizer = dataset.discretizer;  discount = dataset.discount
  observation_dim = dataset.observation_dim;  #** w*h  
  action_dim = dataset.action_dim
  value_fn = lambda x: discretizer.value_expectation(x) 
  timer = Timer()
  #######################
  ###### main loop ######
  #######################
  dargs:DataConfig = dargs
  w_h = dargs.w_h
  maze_state = env_maze.reset();
  obss = maze_state_to_obs(maze_state)
  #***
  rollout = [obss.copy()]## observations for rendering
  context = []## previous (tokenized) transitions for conditioning transformer
  T = timeout
  score_list = []
  for game_id in range(args.simulation_game_num):
    total_reward = 0
    for t in range(T):
      ## concatenate previous transitions and current observations to input to model_
      prefix = make_prefix(discretizer, context, obss, args.prefix_context)
      # # [1,26~196]                    []      (26,)          True
      ## sample sequence_ from model_ beginning with `prefix`
      sequence = beam_plan(dargs,args,
          gpt_model, value_fn, prefix, observation_dim, action_dim, discount)
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
      print(obss.reshape(w_h,w_h),action_map[action_in])
    
      ## update return
      total_reward += reward

      ## update rollout observations and context transitions
      rollout.append(next_obss.copy())
      context = update_context(context, discretizer, obss, action, reward, 
                               args.max_context_transitions,value=sequence_recon[0][-1])
  
      if args.enable_breakpoint: 
        breakpoint()
      print(
          f'[ plan ] t: {t} / {T} | r: {reward:.2f} |  {total_reward=:.2f}  | '
          f'time: {timer():.2f}  | {args.exp_name} | {args.suffix}\n'
      )
      if terminal: break
      obss = next_obss
    maze_state = env_maze.reset()
    obss = maze_state_to_obs(maze_state)
    rollout = [obss.copy()]
    score_list.append(total_reward)
    context=[]
  avg_score = np.array(score_list).mean()
  print(f"====={avg_score=}=====")
  print(score_list)
  return t,total_reward,terminal.item()