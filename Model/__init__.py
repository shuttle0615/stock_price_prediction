MAX_LEN = 100

import torch
import torch.nn as nn
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from .Transformer_model import TransformerEncoder as Model_transformer