import os
import ccxt
from datetime import datetime
import pandas as pd
from Data import cache_dir
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

root = '/home/KyuhoLee/pytorch_tutorial/Stock_prediction_project'

cache_dir = Path(root) / 'cache'
cache_dir.mkdir(parents=True, exist_ok=True)