import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from maze.data.data import VideoDataset

import pyrallis
from config_data import DataConfig
@pyrallis.wrap()    
def main(c: DataConfig):
  data = VideoDataset(c.data_path).get_data()
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

