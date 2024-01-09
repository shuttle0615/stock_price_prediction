import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from .Train import train_loop
from .Validation import validation_loop
