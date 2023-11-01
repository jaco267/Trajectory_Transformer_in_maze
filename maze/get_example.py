import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from maze.data.data import VideoDataset

import pyrallis
from config import TrainConfig
@pyrallis.wrap()    
def main(c: TrainConfig):
  data = VideoDataset(c.data_path,train=True).get_data()
  # print_data_maze(data[11])
  
  print(data.keys())
  print("""
---------type the following code----------

print(data.keys())
print(data['observations'].shape)
        
------------------------------------------
        """)
  breakpoint()
if __name__ == '__main__':
   main()

