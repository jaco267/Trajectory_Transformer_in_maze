import numpy as np
import math
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

from .ein import EinLinear
from z_train.print_maze import get_seq_maze
from config_data import DataConfig
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0  #cfg = emb*head = 32*4 = 128
        # key, query, value projections for all heads
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(cfg.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.resid_pdrop)
        # output projection
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence_
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                                     .view(1, 1, cfg.block_size, cfg.block_size))
        '''
tensor([[[[1., 0., 0.,  ..., 0., 0., 0.],
          [1., 1., 0.,  ..., 0., 0., 0.],
          [1., 1., 1.,  ..., 0., 0., 0.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 0.],
          [1., 1., 1.,  ..., 1., 1., 1.]]]])  
                                (1,1,249,249) = 1,1,seq_len*trans_dim,seq_len*trans_dim
        '''
        #* transition_dim:int = None    #*17+6+1+1 = 25 (obs action,reward,value)
        ##?? mask previous value estimates  #mask value token only keep (obs,action,reward)
        joined_dim = cfg.observation_dim + cfg.action_dim + 2  
        #25 mask value, only keep, obs,action,reward  #?? why
        # mask2 = self.mask.clone()
        self.mask.squeeze()[:,joined_dim-1::joined_dim] = 0   #0:mask, 1:keep
        '''
        self.mask[0][0][200] [24*[1],0, 24*[1],0, ...[0*]*49]
        mask2[0][0][200]  [[1]*200,[0]*49]  
        '''
        self.n_head = cfg.n_head  #4

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        ## [ B x n_heads x T x head_dim ]
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        ## [ B x n_heads x T x T ]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        ## [ B x n_heads x T x head_size ]
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        ## [ B x T x embedding_dim ]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)  #emb*n_head = 32*4=128
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
  """  the full GPT language model, with a context size of block_size """
  def __init__(self, cfg, dargs):
    super().__init__()
    # input embedding stem (+1 for stop token)
    self.dargs:DataConfig = dargs
    self.w_h = dargs.w_h
    #*                                   100 * 25 (obs+action...),   n_emb=n_emb*n_head = 32*4 = 128
    self.tok_emb = nn.Embedding(cfg.vocab_size * cfg.transition_dim + 1, cfg.n_embd)
    self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))  
      #(1,249,128) #249=seq_len*trans_dim-1 = 10*25-1 = 249
    self.drop = nn.Dropout(cfg.embd_pdrop)
    # transformer
    self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])  #n_layer=4
    # decoder head
    self.ln_f = nn.LayerNorm(cfg.n_embd)
    # self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
     #EinLinear: eoi,bei->beo
    self.head = EinLinear(cfg.transition_dim, cfg.n_embd, cfg.vocab_size + 1, bias=False)
   
    self.vocab_size = cfg.vocab_size   #100
    self.stop_token = cfg.vocab_size * cfg.transition_dim  #* 100*25=2500
    self.block_size = cfg.block_size  #seq_len*transdim = 10*25-1 = 249
    self.observation_dim = cfg.observation_dim  #* 17

    self.action_dim = cfg.action_dim  #6
    self.transition_dim = cfg.transition_dim  #17+6+1+1 = obs,action,reward,value = 25
    self.action_weight = cfg.action_weight  #5
    self.reward_weight = cfg.reward_weight  #1
    self.value_weight = cfg.value_weight    #1

    self.embedding_dim = cfg.n_embd  #32*4 = emb_dim*n_head = 128
    self.apply(self._init_weights)
  def get_block_size(self): return self.block_size

  def _init_weights(self, module):
    #** done
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

  def configure_optimizers(self, train_config):
    #** done
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets:    
    those that will experience weight decay for regularization and those that won't 
    (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, EinLinear)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add('pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in self.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
    return optimizer

  def offset_tokens(self, idx):
    #todo
    _, t = idx.shape
    n_states = int(np.ceil(t / self.transition_dim))
    offsets = torch.arange(self.transition_dim) * self.vocab_size
    offsets = offsets.repeat(n_states).to(idx.device)
    offset_idx = idx + offsets[:t]
    offset_idx[idx == self.vocab_size] = self.stop_token
    return offset_idx

  def pad_to_full_observation(self, x, verify=False):
    #todo
    b, t, _ = x.shape
    n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim
    padding = torch.zeros(b, n_pad, self.embedding_dim, device=x.device)
    ## [ B x T' x embedding_dim ]
    x_pad = torch.cat([x, padding], dim=1)
    ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
    x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim)
    if verify:
        self.verify(x, x_pad)
    return x_pad, n_pad

  def verify(self, x, x_pad):
    #todo
    b, t, embedding_dim = x.shape
    n_states = int(np.ceil(t / self.transition_dim))
    inds = torch.arange(0, self.transition_dim).repeat(n_states)[:t]
    for i in range(self.transition_dim):
        x_ = x[:,inds == i]
        t_ = x_.shape[1]
        x_pad_ = x_pad[:,i].view(b, n_states, embedding_dim)[:,:t_]
        print(i, x_.shape, x_pad_.shape)
        try:
            assert (x_ == x_pad_).all()
        except:
            pdb.set_trace()

  def forward(self, idx, targets=None, mask=None):
    """
        idx : [ B x T ]
        values : [ B x 1 x 1 ]
    """
    get_seq_maze(idx,targets, self.w_h,idx=0)  
    b, t = idx.size()  #** (256,249) = (bs,block_size) = (bs, trans_dim*seq_len-1) = (bs,25*10 -1)
    #*** idx.dtype == int64
    assert t <= self.block_size, "Cannot forward, model block size is exhausted."
    offset_idx = self.offset_tokens(idx)
    ''' idx
tensor([[33,  9, 40,  ..., 80, 87, 31], #*(bs,l=249)
        [60, 58, 79,  ...,  3, 23, 65],
        [78, 64, 56,  ..., 45, 34, 65],
        ...,
        [ 9, 94, 71,  ..., 22,  4, 33],
        [97, 28, 76,  ..., 16, 19, 18],
        [55, 49, 94,  ..., 36, 38, 47]], device='cuda:0')
     offset_idx (pos embeding)  #*(bs,l=249)
      ([[  33,  109,  240,  ..., 2180, 2287, 2331],
        [  60,  158,  279,  ..., 2103, 2223, 2365],
        [  78,  164,  256,  ..., 2145, 2234, 2365],
        ...,
        [   9,  194,  271,  ..., 2122, 2204, 2333],
        [  97,  128,  276,  ..., 2116, 2219, 2318],
        [  55,  149,  294,  ..., 2136, 2238, 2347]], device='cuda:0')
    '''

    ## [ B x T x embedding_dim ] = (256,249,128)
    # forward the GPT model
    token_embeddings = self.tok_emb(offset_idx) # each index maps to a (learnable) vector
    # (256,249,128) = (bs,l=249,emb_dim)  #todo  cat the convolution result (ch,emb_dim) here
    #                                           note that conv will also need positional embedding #                                            (maybe you'll want to check how they do it in vit)
    ## [ 1 x T x embedding_dim ] = (1,249,128)
    position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
    ## [ B x T x embedding_dim ] = (256,249,128)
    x = self.drop(token_embeddings + position_embeddings)
    x = self.blocks(x)   
    ## [ B x T x embedding_dim ] = (256,249,128)
    x = self.ln_f(x)
    ## [ (B * T' / transition_dim) x transition_dim x embed_dim ] x_pad= 256*250/25->(2560,25,128)
    x_pad, n_pad = self.pad_to_full_observation(x)   #n_pad = 1
    ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ] 
    logits = self.head(x_pad)
    ## [ B x T' x (vocab_size + 1) ]
    logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
    ## [ B x T x (vocab_size + 1) ] = (256,249,101)
    logits = logits[:,:t]
    # if we are given some desired targets also calculate the loss
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
        if self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
            #### make weights
            n_states = int(np.ceil(t / self.transition_dim))
            weights = torch.cat([
                torch.ones(self.observation_dim, device=idx.device),
                torch.ones(self.action_dim, device=idx.device) * self.action_weight,
                torch.ones(1, device=idx.device) * self.reward_weight,
                torch.ones(1, device=idx.device) * self.value_weight,
            ])
            ## [ t + 1]
            weights = weights.repeat(n_states)
            ## [ b x t ]
            weights = weights[1:].repeat(b, 1)
            ####
            loss = loss * weights.view(-1)
        loss = (loss * mask.view(-1)).mean()
    else:
        loss = None

    return logits, loss
  

# I deleted conditional gpt