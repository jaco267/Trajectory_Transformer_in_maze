from dataclasses import dataclass
N = 100  #dataset_size
device = 'cuda'
n_embd = 32;  n_head = 4
step = 1; subsampled_sequence_length = 10
@dataclass
class Trainer_config:
    batch_size:int = 256
    learning_rate:float=6e-4
    betas:tuple=(0.9, 0.95)
    grad_norm_clip:float=1.0
    weight_decay:float=0.1 # only applied on matmul weights
    # learning rate decay: linear warmup followed by cosine decay to 10% of original
    lr_decay:bool=True
    warmup_tokens:int = None#**warmup_tokens  will be set
    final_tokens:str= None #**final_tokens will be set
    #dataloader
    num_workers:int=0
    device:str=device  #** should be smae as TrainConfig
@dataclass
class GPT_config:
    #* traject.models.transformer.GPT_
    ## discretization
    vocab_size:int=N  
    block_size:int = None  #** subsampled_sequence_length * trans_dim -1 =249
    ## architecture
    n_layer:int = 4;    
    n_head:int = n_head
    n_embd:int = n_embd*n_head   #** n_emb = TrainConfig.n_embd*n_head
    ## dimensions
    observation_dim:int = None;  #*17  
    action_dim:int =None         #*6
    transition_dim:int = None    #*17+6+1+1 = 25 (obs action,reward,done)
    ## loss weighting
    action_weight:int = 5;  reward_weight:int = 1;  value_weight:int  = 1
    ## dropout probabilities
    embd_pdrop:float = 0.1; resid_pdrop:float = 0.1; attn_pdrop:float  = 0.1
@dataclass
class TrainConfig:
    seed:int = 42;  device:str = device;    
    ## number of epochs for a 1M-size datasets; n_epochs = 1M / dataset_size * n_epochs_ref   
    n_epochs_ref:int = 1;  #50    
    n_saves:int = 1; #3
    #** ----datasets----
    savepath:str = 'logs/maze/gpt/pretrained' 
    data_path:str =  'datasets/maze/'
    N:int=N; #* vocab size   
    sequence_length:int = subsampled_sequence_length*step  #10*1=10
    step:int = step;  discount:float = 0.99;
    #** ---transformer---
    n_embd:int = n_embd;
    subsampled_sequence_length:int = subsampled_sequence_length; 
    max_path_length:int = -1 #maze step max_len +subseq_len+1  ex.51 
    #** ---submodule---
    trainer_config:Trainer_config = Trainer_config
    gpt_config:GPT_config = GPT_config
    
    def update_config(self,dataset_len, obs_dim,act_dim, trans_dim):
        #obs=17 , action = 6, joined=17+6+1+1 = 25 
        #10*25 -1 =seq_len*state_joint_dim  
        block_size = self.subsampled_sequence_length * trans_dim -1   
        self.gpt_config.block_size = block_size    #10*25-1 = 249
        self.gpt_config.observation_dim = obs_dim  #17
        self.gpt_config.action_dim = act_dim       #6
        self.gpt_config.transition_dim = trans_dim  #transition_dim = 25
        # joined_dim is in traj/datasets/sequence_.py #joined_dim = obs_dim+act_dim+reward_dim+value_dim
        print(f'Joined dim: {trans_dim} obs: {obs_dim}, action: {act_dim}) | Block: {block_size}' )
        warmup_tokens = dataset_len*block_size
        final_tokens = 20 * warmup_tokens
        self.trainer_config.warmup_tokens = warmup_tokens  #998999*249=len(datasets*block+size)
        self.trainer_config.final_tokens = final_tokens
    def __str__(self):
        fields = [(attribute, value) for attribute, value in self.__dict__.items()]
        for field in fields:  print("{}: {}".format(*field))
        return ""