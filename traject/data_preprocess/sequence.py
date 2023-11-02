import numpy as np
import torch
from utils import discretization
from utils.arrays import to_torch
from maze.data.data import VideoDataset


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
class SequenceDataset(torch.utils.data.Dataset):
  def __init__(self, data_path,  N=50, sequence_length=250, step=10, 
               discount=0.99, max_path_length=11, penalty=None, device='cuda:0'):
    print(f'[datasets/sequence_] Seq len: {sequence_length}, Step: {step}, Max path len: {max_path_length}')       
    print(f'[ datasets/sequence_ ] Loading...', end=' ', flush=True)
    '''
    self.env = env = load_environment(env);self.device = device  #* d4rl
    dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=True) #* d4rl
    '''
    dataset = VideoDataset(data_path).get_data()
    
    '''dataset.keys()
    (['observations', 'actions', 'next_observations', 'rewards', 'terminals', 'realterminals'])
np.ndarray(999999,17) (999999,6)  (999999,17)      (999999,1) (999999,1)     (999999,1)
      fp32                                                         bool          bool
    '''
    # breakpoint()
    self.sequence_length = sequence_length; self.step = step  #10  #1  
    self.max_path_length = max_path_length   #?1000
    obs_b,   act_b, _, rew_b, term_b,_ = get_dataset(dataset)  
    
    # f32     f32        f32   bool   bool    #type(np.ndarray)
    #(bs,17)#(bs,6)    (bs,1)  (bs,1) (bs,1):bool  #bs==999999
    #np.where(term_b.squeeze()==True)  #[999,1999,2999,3999,...998999] #terminal (including timeout)
    #np.where(real_term_b.squeeze()==True) = []   #real_terminal (not include timeout)
    self.joined_raw = np.concatenate([obs_b, act_b], axis=-1)  #(bs,17+6)=(bs,23)
    self.rewards_raw = rew_b              #(bs,1)
    '''
    if penalty is not None:  ## terminal penalty
        terminal_mask = realterm_b.squeeze()   
        self.rewards_raw[terminal_mask] = penalty   #penalty of game over == -100
    '''
    ## segment
    print(f'[ datasets/sequence_ ] Segmenting (obs action)...')
    self.joined_segmented, self.termination_flags, self.path_lengths = segment(
        self.joined_raw, term_b, max_path_length)
    #  (1000,1000,23),  (1000,1000):bool        total tra jnum #[1000,1000,...,999], len==1000
    #  (traj_id, traj_len,dim), term True notTerm False
    print("\nok")
    print(f'[ datasets/sequence_ ] Segmenting (reward)...')
    self.rewards_segmented, *_ = segment(self.rewards_raw, term_b, max_path_length)
    #     (1000,1000,1)  
    print("\nok")
    self.discount = discount  #0.99
    self.discounts = (discount ** np.arange(self.max_path_length))[:,None]
    #      [1,0.99,0.98,....0.01] len(1000)             

    ## [ n_paths x max_path_length x 1 ]
    self.values_segmented = np.zeros(self.rewards_segmented.shape)

    for t in range(max_path_length):
        ## [ n_paths x 1 ]
        # print(t,self.rewards_segmented[:,],self.discounts[:-1])
        #t=0             v0 = r1 + r2*0.99+r3*0.98 +....r1000*0.01
        #t=1             v1 =      r2 + r3*0.99+        n1000*0.011           
        #t=N         v1000 =                                     0
        V = (self.rewards_segmented[:,t+1:] * self.discounts[:-t-1]).sum(axis=1)
        self.values_segmented[:,t] = V
    ## add (r, V) to `joined`
    values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)  #(1000_000)
    # print(values_raw.shape,"vvvv") #(1000_000)
    values_mask = ~self.termination_flags.reshape(-1)  #valuemask --> term is 0 notTerm is 1
    self.values_raw = values_raw[values_mask, None]
    # print(values_raw.shape,">>>>>")     #(1000_000)
    # print(self.values_raw.shape,"vvv")  #(99999,1)   term step is dropped
    #  joined_raw (bs, obs+action) -> (bs,obs+action+rewards+values) = (999999, 17+6+1+1)
    self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
    self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)  ##  (1000,1000,23+1+1) == (1000,1000,25)
    ## get valid indices
    indices = []  #* indice to connect [998999] with (traj_id, traj_len_id) in  __getitem__
    ll = len(self.path_lengths)
    for path_ind, length in enumerate(self.path_lengths):#[1000,1000,...,999], len==1000
        print(f"\r{path_ind}/{ll}",end='')
        end = length - 1
        for i in range(end):
            indices.append((path_ind, i, i+sequence_length))
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
        self.joined_segmented, np.zeros((n_trajectories, sequence_length-1, joined_dim)),
    ], axis=1)  #*(1000,1000+10-1,25)
    self.termination_flags = np.concatenate([
        self.termination_flags, np.ones((n_trajectories, sequence_length-1), dtype=bool),
    ], axis=1)  #* (1000,1000+10-1)
    self.N = N   #100
     # print(self.joined_raw.shape,"hello") #(999999,23+1+1) = (999999,25)
    print("ok?3")
    
    self.discretizer = discretization.QuantileDiscretizer(self.joined_raw, N)
  def __len__(self):    return len(self.indices)
  def __getitem__(self, idx:int):
    ##self.indices[(ind:0~1000,i:0~999,i:10~999+10 ),...()] len(1000000-1001)
    #int         int      int
    path_ind, start_ind, end_ind = self.indices[idx]
    #joined_segmented(1000,1000+10-1,25)  #self.step=1          #step = 1
    joined=self.joined_segmented[path_ind,start_ind:end_ind:self.step]
    #joined=[seq_l,trans_dim]=(10,19)    #              i:(i+seq_len)  i:(i+10)
    #termin_flags  (1000,1000+10-1)
    terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step]
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
    mask[traj_inds > self.max_path_length - self.step] = 0
    ## flatten everything
    joined_discrete = joined_discrete.view(-1)  #(25*10,)
    mask = mask.view(-1)

    X = joined_discrete[:-1]  #src
    Y = joined_discrete[1:]   #tgt
    mask = mask[:-1]

    return X, Y, mask

      
     
      

