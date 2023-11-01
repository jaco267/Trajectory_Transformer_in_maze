from dataclasses import dataclass



@dataclass
class TrainConfig:
  data_path:str = 'datasets/maze/'
  mode:str="gen"  #play or gen
  gen_mode:str="1chan"#"3chan"
  '''
  1chan_flat->obs.shape (w_h,w_h)
        0 1     1 is wall 2 is pos, 4 is goal
        2 4   
  
  3chan -> obs.shape (3,w_h,w_h)
        0 1  | 0 0 | 0 0 
        0 0  | 1 0 | 0 1 
        wall   pos   goal
  '''
  n_steps:int = 3_000  #1000_000  # should be multiple of file_sizes
  file_size:int = 3_000  ## will generate n_steps/(file_size*seq_len)'s file
  # len(data) = file_size*file_num==16_000*1=16_000
  
  n_steps_test:int = 2_000
  file_size_test:int = 2_000
  
  chan:int = 3;   #goal,pos,wall	
  w_h:int = 4;    #resolution
  # print:bool = True
  time_out:int = 10