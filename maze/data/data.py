import os.path as osp
import glob
import torch as tc
import re
# pdb.set_trace()
class VideoDataset():
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    def __init__(self,data_path):
        """data_folder: path to video folder. should contain a 'train' and a 'test' directory, each with corresponding videos stored
        sequence_length: length of extracted video sequences
        """
        super().__init__()
        # pdb.set_trace()
        self.folder = data_path
        files = glob.glob(osp.join(self.folder, '**', f'*.pt'), recursive=True)
        # ['datasets/maze/buf1.pt', 'datasets/maze/buf0.pt']
        self.file_list = list(set([get_parent_dir(f) for f in files]))
        # self.file_list.sort()  #* ['buf0.pt', 'buf1.pt']
        self.file_list = sorted(self.file_list, key=lambda s: int(re.search(r'\d+', s).group()))
    @property
    def file_num(self):  return len(self.file_list)  
    def get_data(self):
      buf_object = tc.load(f"{self.folder}/{self.file_list[0]}")
      buf_obs = buf_object['obs'].numpy()  #[] len n_steps , each with  [3,seq,4,4]
      buf_action = buf_object['action'].numpy()  #[] len n_steps , each with [seq]
      buf_term = buf_object['term'].numpy() 
      buf_reward = buf_object['reward'].numpy()   #* always 1
      buf_t = buf_object['t'].numpy() 
    #   print(f"file_idx: {file_idx} batch_idx {data_idx} {self.file_list[file_idx]} {len(buf_obs)}"
      return_buf_object = {
          'observations': buf_obs, # [10] [3, seq, 4, 4]
          'actions': buf_action,  #[10] [seq]
          'terminals':buf_term,
          'rewards':buf_reward,
          't':buf_t,
      }
      return return_buf_object  #{obs(ch,t,4,4), action(bs)}
def get_parent_dir(path): return osp.basename(path)