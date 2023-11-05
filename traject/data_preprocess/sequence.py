import numpy as np
import torch
from utils import discretization
from utils.arrays import to_torch
from maze.data.data import VideoDataset
from config_data import DataConfig
from config_train import TrainConfig
Action_dict = {  0: 'stay',1: 'up',2: 'left',3: 'down',4: 'right'}
def segment(observations,  #(bs,17+6)=(bs,23)=(bs,obs+action)  #bs==999999
            term_b,     #(bs,1)
            max_path_length  #1000
    ):
    """segment `observations` into trajectories according to `terminals`"""
    #np.where(term_b.squeeze()==True)  #[999,1999,2999,3999,...998999] len(term_b) = 999999
    assert len(observations) == len(term_b)
    observation_dim = observations.shape[1]  #23
    trajectories = [[]]  # shape(bs,23)
    len_obs = len(observations)
    for i, (obs, term) in enumerate(zip(observations, term_b)):
        print(f"\r{i}/{len_obs}",end='')
        trajectories[-1].append(obs)
        #term == done or timeout
        if term.squeeze():  #if done  , start new episode
            trajectories.append([])  # go to the next episode
    # trajectorys shape~= [1000,1000] = (traj_id, traj_len)  --> [1000,1000,1000,....1000,999]
    # np.array(trajectories[0:-1]).shape == (999,1000,23)    len(trajectories[-1]) = 999
    
    if len(trajectories[-1]) == 0: trajectories = trajectories[:-1]
    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]  #[(1000,23)]*999+[(999,23)]*1
    
    n_trajectories = len(trajectories)   #1000
    path_lengths = [len(traj) for traj in trajectories]  #[1000,1000,...,999]
    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros((n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype)  #(1000,1000,23)
    #                             (1000,1000)
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]  #1000   or 999 in last i 
        trajectories_pad[i,:path_length] = traj
        early_termination[i,path_length:] = 1   # [[0]*1000]*999 + [[0]*999+[1]]*1
        #np.where(early_termination.squeeze()==True)= (999,999)
    #     (1000,1000,23)    (1000,1000):bool        total tra jnum #[1000,1000,...,999], len==1000
    return trajectories_pad, early_termination, path_lengths   
def get_dataset(dataset):
    obs_b = dataset['observations']  #obs_batch
    obs_flat_b = obs_b.reshape(obs_b.shape[0],-1)  #(bs,w,h) ->(bs,w*h)
    act_b = dataset['actions'].reshape(-1,1)
    next_obs_b = None  # we dont need this
    rew_b = dataset['rewards'].reshape(-1,1)
    term_b = dataset['terminals'].reshape(-1,1)   #done or timeout
    realterm_b =None  #only done
    # breakpoint()
    print("shapes:  ",obs_flat_b.shape,act_b.shape,
           np.concatenate([obs_flat_b, act_b], axis=-1).shape ,rew_b.shape,
           term_b.shape,">>>>")
    return obs_flat_b, act_b,next_obs_b,rew_b,term_b,realterm_b
def print_maze(obs,act,term,start,end):
   w_h = int(obs.shape[-1]**0.5)
   for i in range(start,end):
      print(obs[i].reshape(w_h,w_h),Action_dict[int(act[i].item())],term[i]),
class SequenceDataset(torch.utils.data.Dataset):
  def __init__(self,dargs,targs):
    self.dargs:DataConfig = dargs;  self.targs:TrainConfig = targs
    dataset = VideoDataset(self.targs.data_path).get_data()
    self.seq_len = self.targs.sequence_length ; #10 
    self.step = self.targs.step    #1  
    
    
    max_path_len=targs.max_path_length if targs.max_path_length is not -1 else self.dargs.time_out+1
    self.max_path_len = max_path_len  #time_out + 1
    self.discount=self.targs.discount  #0.99
    self.discounts = (self.discount ** np.arange(self.max_path_len))[:,None]
    #      [1,0.99,0.98,....0.01] len(time_out+1)   
    print(f'Seq len: {self.seq_len}, Step: {self.step}, Max path: {self.max_path_len}')      
    obs_b,   act_b, _, rew_b, term_b,_ = get_dataset(dataset)  
    #(bs,w_h**2) (bs,1)  (bs,1)  (bs,1)bool 
    #print_maze(obs_b,act_b,term_b,start=45,end=51)     game1 g2 g3 g4 ...
    #print(np.where(term_b.squeeze()==True)[0][:10])   #[ 50  51  54 105 129 134 185 236 260 267]
    self.joined_raw = np.concatenate([obs_b, act_b], axis=-1)  #(bs,w_h^2+1)
    self.rewards_raw = rew_b              #(bs,1)
    print(f'[ datasets/sequence_ ] Segmenting (obs action)...')
    self.joined_segmented,self.term_flags,self.path_lens=segment(self.joined_raw,term_b,max_path_len)
    #(eps_num,max_path_len,w_h^2+1),  (eps_num,max_path_len):bool, [51,51,3,38,19...51], len=eps_num
    print(f'[ datasets/sequence_ ] Segmenting (reward)...')
    self.rewards_segmented, *_ = segment(self.rewards_raw, term_b, max_path_len)
    #(eps_num,max_path_len,1)  
    self.values_segmented = np.zeros(self.rewards_segmented.shape)
    #(eps_num,max_path_len,1) 
    
    for t in range(max_path_len):
        ## [ n_paths x 1 ]
        # print(t,self.rewards_segmented[:,],self.discounts[:-1])
        #t=0             v0 = r1 + r2*0.99+r3*0.98 +....r51*0.01
        #t=1             v1 =      r2 + r3*0.99+        r51*0.011           
        #t=N            v51 =                                  0
        V = (self.rewards_segmented[:,t+1:] * self.discounts[:-t-1]).sum(axis=1)
        self.values_segmented[:,t] = V
    ## add (r, V) to `joined`
    values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)  #(eps_num*max_path_len,)
    
    values_mask = ~self.term_flags.reshape(-1)  #valuemask --> term is 0 notTerm is 1
    self.values_raw = values_raw[values_mask, None]  #(bs,1)  #ex. (1000_000,1)
    # term step is dropped
    #  joined_raw (bs, obs+action) -> (bs,obs+action+rewards+values) = (bs, w_h^2+1+1+1)
    self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
    self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)  ##  (eps_id,time_out+1,w_h^2+1+1+1) 
    ## get valid indices
    indices = []  #* indice to connect [bs] with (traj_id, traj_len_id) in  __getitem__
    ll = len(self.path_lens)
    for path_ind, length in enumerate(self.path_lens):#[1000,1000,...,999], len==1000
        print(f"\r{path_ind}/{ll}",end='')
        end = length - 1
        for i in range(end):
            indices.append((path_ind, i, i+self.seq_len))
    # print(indices)  #[(ind:0~999,i:0~998,i:10~999+10 ),(),...] len(1000000-1001)
    # print(len(indices),"wejkejrkejrk")
    self.indices = np.array(indices)  #999*999 + 998 - 998999,  self.indices.shape= (998999,3)
    self.observation_dim = obs_b.shape[1]  #17
    self.action_dim = act_b.shape[1]            #6
    self.joined_dim = self.joined_raw.shape[1]    #17+6=23
    ## pad trajectories
    n_trajectories, _, joined_dim = self.joined_segmented.shape #[1000,1000,25]
    #*** this is what the __getitem__ get
    self.joined_segmented = np.concatenate([
        self.joined_segmented, np.zeros((n_trajectories, self.seq_len-1, joined_dim)),
    ], axis=1)  #*(eps_id,timeout+seq_len,25)
    self.term_flags = np.concatenate([
        self.term_flags, np.ones((n_trajectories, self.seq_len-1), dtype=bool),
    ], axis=1)  #* (eps_id,timeout+seq_len)
    self.N = self.targs.N   #100
     # print(self.joined_raw.shape,"hello") #(999999,23+1+1) = (999999,25)
    self.discretizer = discretization.QuantileDiscretizer(self.dargs, self.joined_raw, self.N)
  def __len__(self):    return len(self.indices)
  def __getitem__(self, idx:int):
    ##self.indices[(ind:0~1000,i:0~999,i:10~999+10 ),...()] len(1000000-1001)
    #int         int      int
    path_ind, start_ind, end_ind = self.indices[idx]
    #joined_segmented(eps_id,timeout+seq_len,w_h^2+3)  #self.step=1          #step = 1
    joined=self.joined_segmented[path_ind,start_ind:end_ind:self.step]
    #joined=[seq_l,trans_dim]=(seq_l,w_h^2+3)    #i:(i+seq_l)  i:(i+10)

    #termin_flags  (eps_id,timeout+seq_len)
    terminations = self.term_flags[path_ind, start_ind:end_ind:self.step]
    #term = (10,)  # [False]*10 
    joined_discrete = self.discretizer.discretize(joined)  #discretized along each dimension
    # joined_discrete = joined.astype(np.int64) #todo what about the continues value

    # np.set_printoptions(suppress = True)
    #print(1+joined[:,0])    #[0.8712, 0.897, 0.94, 0.96, 0.91, 0.879, 0.852, 0.872, 0.871, 0.895 ]
    #print(joined_discrete[:,0])#[ 15,    27,   56,   69,   41,    18,     9,    15,    15,    26 ]
    #(10,25)
    ## replace with termination token if the sequence_ has ended
    assert (joined[terminations] == 0).all(), \
            f'Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}'
    # print(joined_discrete[0])
    # (10,25)[(10,)]
    # if terminations.any()==True:  #everything in term is 1000
    #     print(joined_discrete[0])
    #     breakpoint()

    ## [ (sequence_length / skip) x observation_dim]
    joined_discrete = to_torch(joined_discrete, device='cpu', dtype=torch.long).contiguous()
    #(10,25)
    
    ## don't compute loss for parts of the prediction that extend
    ## beyond the max path length
    #                        0~999      0~999+10
    traj_inds = torch.arange(start_ind, end_ind, self.step)
    mask = torch.ones(joined_discrete.shape, dtype=torch.bool)  #mask(10,25) #
    mask[traj_inds > self.max_path_len - self.step] = 0
    ## flatten everything
    joined_discrete = joined_discrete.view(-1)  #(25*10,)
    mask = mask.view(-1)

    X = joined_discrete[:-1]  #src  #seq_l * trans_dim = (10*(w_h^2+1+1+1)
    Y = joined_discrete[1:]   #tgt
    mask = mask[:-1]
    return X, Y, mask

      
     
      

