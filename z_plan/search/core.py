from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from my_conf_maze.utils.discretization import QuantileDiscretizer

import numpy as np
import torch

from .sampling import sample_n, get_logp, sort_2d
from config_data import DataConfig
from config_plan import PlanConfig
REWARD_DIM = VALUE_DIM = 1

action_map = {
   0:'no_op',
   1:'up',
   2:'left',
   3:'down',
   4:'right'
}
def to_maze(x):
    w_h_2 = x.shape[0] -1 

    w_h = int(w_h_2**0.5)
    print(x[:w_h_2].reshape(w_h,w_h),x[w_h_2].item(),action_map[x[w_h_2].item()])

@torch.no_grad()
def beam_plan(dargs,args, model, value_fn, x,
    observation_dim, action_dim,
    discount=0.99, 
):
    '''x : tensor[ 1 x input_sequence_length ]'''
    dargs:DataConfig = dargs
    args:PlanConfig = args
    k_obs = args.k_obs
    cdf_obs = args.cdf_obs
    verbose = args.verbose
    n_expand = args.n_expand #2
    beam_width = args.beam_width #64
    horizon = args.horizon  #5
    max_context_transitions = args.max_context_transitions
    w_h = dargs.w_h


    warv = w_h**2 + 3 # w_h**2, action,reward value
    inp = x.clone()
    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None
    ## pass in max numer of tokens to sample function
    sample_kwargs = {'max_block': max_block, 'crop_increment': transition_dim, }
    ## repeat input for search
    x = x.repeat(beam_width, 1)  #(64,16) = (beam_width , obs)
    ## construct reward and discount tensors for estimating values
    rewards = torch.zeros(beam_width, horizon + 1, device=x.device)
    discounts = discount ** torch.arange(horizon + 1, device=x.device)

    '''
    step 0                         1                        2 3 4 
     obs action reward value   obs action reward value  ,.....
      17   6        1     1     17  6       1      1
           23             25    42  48             50
        trans_dim == 25          trans_dim == 25     
    '''
    for t in range(horizon):
        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)  #(64*2,16)
        rewards = rewards.repeat(n_expand, 1)
        ## sample actions          action_dim=1
        x, rrr_prob = sample_n(model, x, action_dim,mode='action', topk=None, cdf=None, **sample_kwargs)

        # (128,16+1)  #obs,action
        ## sample reward and value estimate
        x, r_probs = sample_n(model, x, REWARD_DIM + VALUE_DIM, topk=None, cdf=None, **sample_kwargs)
        # (128,16+1+1+1)  #obs,action,reward, value

        ## optionally, use a percentile or mean of the reward and
        ## value distributions instead of sampled tokens
        r_t, _ = value_fn(r_probs)
        # QuantileDiscretizer.value_expectation
        # -r_probs[:,0,:]@torch.arange(101,dtype=torch.float32,device='cuda')
        # print(torch.where(~torch.isclose(r_t,torch.tensor(-1,dtype=torch.float32))))
        V_t = -r_probs[:,1,:]@torch.arange(101,dtype=torch.float32,device='cuda')
        assert V_t.shape == r_t.shape
        ## update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)
        ## get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)
        ## index into search candidates to retain `beam_width` highest-reward sequences
        # breakpoint() 
        x = x[inds]
        rewards = rewards[inds]
        ## sample next observation (unless we have reached the end of the planning horizon)
        # breakpoint() #x[:,-3]
        if t < horizon - 1:
            x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)
    
        ## logging
        print(f"\rx:{list(x.shape)} vmin {values.min()} vmax: {values.max()}\
    # vtmin {V_t.min()} vtmax {V_t.max()} discount {discount}",end='')
    if verbose:
      for i in range(horizon):
        to_maze(x[0,warv*i:warv*(i+1)-2])
    # progress.stamp()
    
    ## [ batch_size x (n_context + n_steps) x transition_dim ]
    x = x.view(beam_width, -1, transition_dim)

    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -horizon:]

    ## return best sequence_
    argmax = values.argmax()  #argmax
    best_sequence = x[argmax]
    return best_sequence

@torch.no_grad()
def beam_search(model, x, n_steps, beam_width=512, goal=None, **sample_kwargs):
    batch_size = len(x)

    prefix_i = torch.arange(len(x), dtype=torch.long, device=x.device)
    cumulative_logp = torch.zeros(batch_size, 1, device=x.device)

    for t in range(n_steps):

        if goal is not None:
            goal_rep = goal.repeat(len(x), 1)
            logp = get_logp(model, x, goal=goal_rep, **sample_kwargs)
        else:
            logp = get_logp(model, x, **sample_kwargs)

        candidate_logp = cumulative_logp + logp
        sorted_logp, sorted_i, sorted_j = sort_2d(candidate_logp)

        n_candidates = (candidate_logp > -np.inf).sum().item()
        n_retain = min(n_candidates, beam_width)
        cumulative_logp = sorted_logp[:n_retain].unsqueeze(-1)

        sorted_i = sorted_i[:n_retain]
        sorted_j = sorted_j[:n_retain].unsqueeze(-1)

        x = torch.cat([x[sorted_i], sorted_j], dim=-1)
        prefix_i = prefix_i[sorted_i]

    x = x[0]
    return x, cumulative_logp.squeeze()
