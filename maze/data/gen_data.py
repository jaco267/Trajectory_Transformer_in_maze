from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import TrainConfig
import torch as tc
from data.utils import chan
import numpy as np
import os

def get_init_buffer(c:TrainConfig,n_steps):
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
def generate_maze_data(c:TrainConfig,env,mode='train'):
  '''generate data of [(bs,chan,seq_len,h,w)]*file_siz in each file'''
  train_folder = c.data_path+"train/";  	  test_folder = c.data_path+"test/"
  os.makedirs(train_folder,exist_ok=True);  os.makedirs(test_folder,exist_ok=True)
  
  save_folder = train_folder if mode == 'train' else test_folder
  n_steps       = c.n_steps    if mode == 'train' else c.n_steps_test

  num_actions = env.num_actions()
  buf_obs, buf_action, buf_term, buf_reward, buf_t  = get_init_buffer(c,n_steps)
  
  
  data_idx = 0  
  step = 0
  file_local_idx = data_idx%n_steps
  env_state_bs = env.reset() 
  terminal = tc.tensor(False) 
  while step <= n_steps: 
    while not terminal:
      # print(f"\rstep {step}/{n_steps} file_local_idx {file_local_idx}",end='')
      
      goal, wall_grid, pos, t = env_state_bs 
      # a=(goal:(bs,2)int64,wall_grid:(bs,h_w,h_w)bool,pos:(bs,2)int64,t:(32)int64)
      if c.gen_mode == "3chan":
        set_buf_obs_3chan(buf_obs, file_local_idx, goal,wall_grid,pos)
      elif c.gen_mode == "1chan":
        set_buf_obs_1chan(buf_obs, file_local_idx, goal,wall_grid,pos)
      else:    raise Exception('not implement error...')  
      action = np.random.randint(num_actions) #todo del act0 (stay)
      env_state_bs, _ ,reward, terminal, _ = env.step(action,env_state_bs)  
      buf_action[file_local_idx] = action   #*dtype=tc.int64) 
      buf_term[file_local_idx] = terminal
      buf_reward[file_local_idx] = reward
      buf_t[file_local_idx] = t
      step += 1
      data_idx+=1; 
      file_local_idx = data_idx%n_steps
      if data_idx>n_steps:
        breakpoint()
        print(data_idx,".....")
      if file_local_idx == 0:
        #*  finished one run, save result to folder and initialize buffer
        buf_object= { 'obs':buf_obs,  'action':buf_action, 'term':buf_term, 
                     'reward':buf_reward, 't':buf_t  }
        # breakpoint()
        tc.save(buf_object,save_folder+f"buf{data_idx//n_steps -1}.pt") ## -1 to start from zero
        
        buf_obs, buf_action, buf_term, buf_reward, buf_t = get_init_buffer(c, n_steps)
    env_state_bs = env.reset()  
    terminal = tc.tensor(False)