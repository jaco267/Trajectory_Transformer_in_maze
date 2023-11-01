import os
import cloudpickle as pickle   #we use cloudpickle because pickle cant store dataclase properly
# import pickle
def savedata_config(savepath,args):
    savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
    pickle.dump(args, open(savepath, 'wb'))
    print(f'Saved config to: {savepath}\n')