import numpy as np
import torch

from utils.arrays import to_torch


def make_prefix(discretizer, context, obs, prefix_context=True):
    obs_discrete = discretizer.discretize(obs, only_obs=True)
    obs_discrete = to_torch(obs_discrete, dtype=torch.long)

    if prefix_context:
        prefix = torch.cat(context + [obs_discrete], dim=-1)
    else:
        prefix = obs_discrete

    return prefix

def extract_actions(x, observation_dim, action_dim, t=None):
    assert x.shape[1] == observation_dim + action_dim + 2
    actions = x[:, observation_dim:observation_dim+action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions

def update_context(context, discretizer, observation, 
                   action, reward, max_context_transitions,value):
    '''
        context : list of transitions
            [ tensor( transition_dim ), ... ]
    '''
    ## use a placeholder for value because input values are masked out by model
    rew_val = np.array([reward, value])  #this is just because we dont know value
    transition = np.concatenate([observation, action, rew_val])
    # breakpoint()
    ## discretize_ transition and convert to torch tensor
    transition_discrete = discretizer.discretize(transition) 
    transition_discrete = to_torch(transition_discrete, dtype=torch.long)

    ## add new transition to context
    context.append(transition_discrete)

    ## crop context if necessary
    context = context[-max_context_transitions:]
    return context