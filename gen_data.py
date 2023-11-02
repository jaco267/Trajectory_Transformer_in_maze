from maze.env_maze.env import ProcMaze
from maze.env_maze.utils import play_with_env

from maze.data.data import VideoDataset
from maze.data.gen_data import generate_maze_data
from maze.data.utils import print_data_maze
from utils.save import savedata_config
import pyrallis
from config_data import DataConfig
@pyrallis.wrap()    
def main(c: DataConfig):
  print(c)
  
  env = ProcMaze(grid_size=c.w_h,device='cpu',timeout=c.time_out)
  if c.mode == "play":	play_with_env(env)
  if c.mode == "gen": 
    generate_maze_data(c,env)
    dataset = VideoDataset(c.data_path)
  elif c.mode == "view_train":
    dataset = VideoDataset(c.data_path)
  else:
    raise Exception('dont have this option...') 
  # print(data[0])
  data = dataset.get_data()  #{obs(bs,ch,4,4), action(bs)}
  # data[11]
  # data[19]
  savedata_config(savepath=(c.data_path,'data_config.pkl'),args=c)
  print_data_maze(c,data,idx=11)
  print("""
---------type the following code----------
        
print_data_maze(c,data,idx=12)
        
------------------------------------------
        """)
  breakpoint()
if __name__ == '__main__':
   main()

