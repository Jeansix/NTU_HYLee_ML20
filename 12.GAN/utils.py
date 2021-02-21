import random
import torch
import numpy as np


# for the convenience of reproduce
def same_seeds(seed):
    """
       固定训练的随机种子，以便reproduce
       :param seed:当前的种子
     """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
