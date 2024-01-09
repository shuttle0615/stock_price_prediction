from args import args

MAX_LEN = 100

import torch
import torch.nn as nn
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'