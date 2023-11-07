import torch 
Action_dict = {  0: 'stay',1: 'up',2: 'left',3: 'down',4: 'right'}
def get_seq_maze(seq_data_X,seq_data_Y,w_h,idx):
   trans_dim =int( w_h**2 + 1+1+1)
   if seq_data_Y == None:  #evaluation
      seq_data_raw = seq_data_X
      #25  26 .....  #shape should be multiple of obs_dim+action_dim+reward_dim+value_dim
      if seq_data_raw.shape[-1] % trans_dim != 0:    
        return   #print(seq_data_X[0].shape)
   else:                   #training
      seq_x = seq_data_X[idx]  #(seq_len*trans_dim -1,)
      seq_y = seq_data_Y[idx]  #(seq_len*trans_dim -1,)
      seq_data_raw = torch.cat((seq_x,seq_y[-1].reshape(1))) ##(seq_len*trans_dim,)
   assert seq_data_raw.shape[-1] % trans_dim == 0
   seq_data = seq_data_raw.reshape(-1,trans_dim)
   seq_len = seq_data.shape[0]
   for trans in seq_data:
      print(trans[0:-3].reshape(w_h,w_h),"action:",Action_dict[trans[-3].item()],
            "value:",-trans[-1].item())