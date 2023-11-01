import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from env_maze.env import ProcMaze
from env_maze.utils import play_with_env

from maze.data.data import VideoDataset
from data.gen_data import generate_maze_data
from data.utils import print_data_maze

import pyrallis
from config import TrainConfig
@pyrallis.wrap()    
def main(c: TrainConfig):
  print(c)
  env = ProcMaze(grid_size=c.w_h,device='cpu',timeout=c.time_out)
  if c.mode == "play":	play_with_env(env)
  if c.mode == "gen": 
    print("\n====gernerating train data====")
    generate_maze_data(c,env,mode='train')
    print("\n====gernerating test data====")
    generate_maze_data(c,env,mode='test')
    dataset = VideoDataset(c.data_path,train=True)
  elif c.mode == "gen_test":
    print("\n====gernerating test data====")
    generate_maze_data(c,env,mode='test')
    dataset = VideoDataset(c.data_path,train=False)
  elif c.mode == "view_train":
    #len(data) == 10000
    dataset = VideoDataset(c.data_path,train=True)
    #len(data) == 10_000
  elif c.mode == "view_test":
    #len(data) == 1000
    dataset = VideoDataset(c.data_path,train=False)
  else:
    raise Exception('dont have this option...') 
  # print(data[0])
  data = dataset.get_data()  #{obs(bs,ch,4,4), action(bs)}
  # data[11]
  # data[19]
  print_data_maze(c,data,idx=11)
  print("""
---------type the following code----------
        
print_data_maze(c,data,idx=12)
        
------------------------------------------
        """)
  breakpoint()
if __name__ == '__main__':
   main()

