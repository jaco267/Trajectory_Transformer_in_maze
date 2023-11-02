from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config_data import DataConfig
import torch as tc
from maze.data.utils import chan
import numpy as np
import os

def get_init_buffer(c:DataConfig,n_steps):
  if c.gen_mode == "3chan":
    # 3 channel [goal,pos,wall]	
    buf_obs    = tc.zeros(n_steps, 3, c.w_h, c.w_h, dtype=tc.float32) 
  elif c.gen_mode == "1chan":
    buf_obs    = tc.zeros(n_steps, c.w_h , c.w_h, dtype=tc.float32) 
  else:    raise Exception('not implement error...')  
  buf_action = tc.zeros(n_steps,dtype=tc.float32)  
  #* you can use int, but we need to merge it with obs in datasets/seq   
  buf_term   = tc.ones (n_steps,dtype=tc.bool) #terminate
  buf_reward = tc.zeros(n_steps,dtype=tc.float32)
  buf_t      = tc.zeros(n_steps,dtype=tc.int32)  #t_step
  return buf_obs, buf_action, buf_term, buf_reward,buf_t
def set_buf_obs_3chan(buf_obs,file_local_idx,goal,wall_grid,pos):
  wall_grid = wall_grid.to(tc.float32)  
  buf_obs[file_local_idx,chan.goal,goal[0],goal[1]] = 1# bs, chan=3, w,h
  buf_obs[file_local_idx,chan.pos ,pos[0] ,pos[1]]  = 1 
  buf_obs[file_local_idx,chan.wall_grid ,:,:]  = wall_grid
def set_buf_obs_1chan(buf_obs,file_local_idx,goal,wall_grid,pos):
  wall_grid = wall_grid.to(tc.float32)  
  #                       bs, chan=3, t=8, w,h
  buf_obs[file_local_idx,goal[0],goal[1]] += 4  #goal is 4
  buf_obs[file_local_idx,pos[0] ,pos[1]]  += 2  #pos is 2
  buf_obs[file_local_idx,:,:]  += wall_grid   #wall_grid is 1
def generate_maze_data(c:DataConfig,env):
  '''generate data of [(bs,chan,seq_len,h,w)]*file_siz in each file'''
  data_folder = c.data_path 	
  os.makedirs(data_folder,exist_ok=True);  
  n_steps       = c.n_steps 

  num_actions = env.num_actions()
  buf_obs, buf_action, buf_term, buf_reward, buf_t  = get_init_buffer(c,n_steps)
  
  step = 0
  env_state_bs = env.reset() 
  terminal = tc.tensor(False) 
  flag = 0
  while step <= n_steps: 
    while not terminal:
      print(f"\rstep {step}/{n_steps}",end='')
      goal, wall_grid, pos, t = env_state_bs 
      # a=(goal:(bs,2)int64,wall_grid:(bs,h_w,h_w)bool,pos:(bs,2)int64,t:(32)int64)
      if c.gen_mode == "3chan":
        set_buf_obs_3chan(buf_obs, step, goal,wall_grid,pos)
      elif c.gen_mode == "1chan":
        set_buf_obs_1chan(buf_obs, step, goal,wall_grid,pos)
      else:    raise Exception('not implement error...')  
      action = np.random.randint(num_actions) #todo del act0 (stay)
      env_state_bs, _ ,reward, terminal, _ = env.step(action,env_state_bs)  
      buf_action[step] = action   #*dtype=tc.int64) 
      buf_term[step] = terminal
      buf_reward[step] = reward
      buf_t[step] = t
      assert step < n_steps
      step += 1
      
      if step == n_steps:
        buf_object= { 'obs':buf_obs,  'action':buf_action, 'term':buf_term, 
                     'reward':buf_reward, 't':buf_t  }
        tc.save(buf_object,data_folder+f"buf{step//n_steps -1}.pt") ## -1 to start from zero
        flag=1
        break
    if flag == 1:
      break
    env_state_bs = env.reset()  
    terminal = tc.tensor(False)