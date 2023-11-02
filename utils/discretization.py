import numpy as np
import torch
import pdb

from .arrays import to_np, to_torch

class QuantileDiscretizer:
	def __init__(self, data, N):
		self.data = data   # (999999,17+6+1+1) = (999999,25)  
		self.N:int = N     #100
		n_points_per_bin = int(np.ceil(len(data) / N)) #10000 = 999999/100
		obs_sorted = np.sort(data, axis=0)  #(999999,25)
		thresholds = obs_sorted[::n_points_per_bin, :]    #(100,25)=(N,25)
		#  so that is the theshold from small to large
		maxs = data.max(axis=0, keepdims=True) #(1,25) float
		## [ (N + 1) x dim ] = (101,25)
		self.thresholds = np.concatenate([thresholds, maxs], axis=0)
		## [ N x dim ]
		# a = t[:-1] =   [1,2,5,6]  8
		# b = t[1:]  = 1 [2,5,6,8]                b-a =  1,3,1,2   
		self.diffs = self.thresholds[1:] - self.thresholds[:-1]
	def discretize(self, x, only_obs=False):
		assert torch.is_tensor(x) == False  #should be np array
		if x.ndim == 1:  x = x[None]## enforce batch mode
		if only_obs:
			assert x.shape[-1] == 16  #obs(16)
			assert np.all(x>=0)
		else: 
			assert x.shape[-1] == 19  #obs(16),action(1),reward(1),value(1)
			assert np.all(x[:,:16]>=0)
			# print(x)
			assert np.all(x[:,17:]<=0)  
		#** called by traject/datasets/sequence_.py self.discretizer.discretiz
		#x(10,19) = (seq_len,trans_dim)
		
		
		## [ N x B x observation_dim ]
		  #(1000,25)->(1,1000,25), (101,25) -> (101,1,25)
		x2 = x.copy().astype(np.int64)  
		#??x2 sometimes will have 1e06 very large value 
		#?? pretty sure that its a bug 
		if only_obs == False:
			x2[:,-2:] = -x2[:,-2:]
		if x2.min() < 0 or x2.max() >= self.N:
			x2 = np.clip(x2, 0, self.N - 1)
		return x2 #indices  #x.astype(np.int64)
	def reconstruct(self, indices):
		#* called at plan reconstruct from output (token to value)
		#  remove this (todo)
		if torch.is_tensor(indices): indices = to_np(indices)
		assert indices.shape[-1] == 19
		assert len(indices.shape) == 2
		assert np.all(indices>=0)
		
		recon = indices.copy().astype(np.float32)
		recon[:,-2:] = -recon[:,-2:]
		#**  todo  analysis transformer output
		return recon
	#---------------------------- wrappers for planning ----------------------------#
	def expectation(self, probs, subslice):
		'''probs : [ B x N ]'''
		if torch.is_tensor(probs):
			probs = to_np(probs)
		## [ N ]
		thresholds = self.thresholds[:, subslice]
		## [ B ]
		left  = probs @ thresholds[:-1]
		right = probs @ thresholds[1:]
		avg = (left + right) / 2.
		# if probs[0,1] <0.9 and probs[0,1]<0.01:
		# 	breakpoint()
		return avg
	#---------------------------- wrappers for planning ----------------------------#

	def value_expectation(self, probs):
		'''
			probs : [ B x 2 x ( N + 1 ) ]
				extra token comes from termination
		'''
		
		if torch.is_tensor(probs):
			probs = to_np(probs)
			return_torch = True
		else:
			return_torch = False

		probs = probs[:, :, :-1]
		assert probs.shape[-1] == self.N

		rewards = self.expectation(probs[:, 0], subslice=-2)
		next_values = self.expectation(probs[:, 1], subslice=-1)

		if return_torch:
			rewards = to_torch(rewards)
			next_values = to_torch(next_values)

		return rewards, next_values

