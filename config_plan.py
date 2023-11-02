
from dataclasses import dataclass
from traject.utils import watch
import os
from traject.utils.serialization import mkdir
logbase = 'logs/'
args_to_watch = [
    ('prefix', ''),
    ('plan_freq', 'freq'),
    ('horizon', 'H'),
    ('beam_width', 'beam'),
]
@dataclass
class PlanConfig:
    gpt_epoch:str = "latest";    
    device:str = "cuda"; 
    seed:int = None;
    plan_freq:int = 1
    horizon:int = 4#15   #?? 15 is two long and it will fail
    beam_width:int = 24#128
    n_expand:int = 2
    enable_breakpoint:bool = True
    k_obs:int = 1
    k_act:int = None
    cdf_obs:int = None
    cdf_act:float = None#0.6
    percentile:str = "mean"

    max_context_transitions:int = 5
    prefix_context:bool = True

    vis_freq:int = 50
    exp_name:str = watch(args_to_watch)  
    prefix:str = 'plans/defaults/'
    suffix:str = '0'
    verbose:bool = True
    savepath:str = None  #post _init 
    def __post_init__(self):
        self.exp_name = self.exp_name(self)  #plans/defaults/freq1_H5_beam32
        self.savepath = os.path.join('logs/maze',self.exp_name,self.suffix)
        #logs/maze/plans/defaults/freq1_H5_beam32/0
        if mkdir(self.savepath): print(f'[ utils/setup ] Made savepath: {self.savepath}')
