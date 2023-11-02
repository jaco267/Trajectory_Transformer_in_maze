from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config_data import DataConfig
from dataclasses import dataclass
@dataclass
class Channel:
   goal:int = 0
   pos:int  = 1
   wall_grid:int = 2

chan = Channel()
Action_dict = {  0: 'stay',1: 'up',2: 'left',3: 'down',4: 'right'}
import numpy as np
def merge_maze_channel(raw_maze_map):
   maze = np.zeros_like(raw_maze_map[0])
   maze += raw_maze_map[chan.wall_grid]
   maze += 3*raw_maze_map[chan.pos]
   maze += 6*raw_maze_map[chan.goal]
   return maze

def print_data_maze(c:DataConfig,buf,idx):
   
   obs = buf['observations'][idx]; 
   action = buf['actions'][idx];
   term = buf['terminals'][idx]; 
   reward = buf['rewards'][idx]; 
   t = buf['t'][idx]
   action = Action_dict[action.item()]
   maze = None
   if c.gen_mode == '3chan':
      maze = merge_maze_channel(obs)
   elif c.gen_mode == '1chan':
      maze = obs 
   else: raise Exception('error...') 
   print(f"{maze} action {action}, terminate {term}, reward {reward}, t {t}")