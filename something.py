import ccxt
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
import pandas as pd
from datetime import datetime

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)