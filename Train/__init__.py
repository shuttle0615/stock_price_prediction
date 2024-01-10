import torch
from pathlib import Path
from Data import root

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

result_dir = Path(root) / 'results'
result_dir.mkdir(parents=True, exist_ok=True)

from .Train import train_loop
from .Validation import validation_loop
from .Experiment import experiment
