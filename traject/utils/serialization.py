import os
def mkdir(savepath, prune_fname=False):
    """
        returns `True` iff `savepath` is created
    """
    if prune_fname:
        savepath = os.path.dirname(savepath)
    if not os.path.exists(savepath):
        try:
            os.makedirs(savepath)
        except:
            print(f'[ utils/serialization ] Warning: did not make directory: {savepath}')
            return False
        return True
    else:
        return False

