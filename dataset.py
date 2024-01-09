import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from Datareader.binance.Binance_data_reader import Data_reader
import pandas as pd
import os
from pathlib import Path

root = "/home/KyuhoLee/pytorch_tutorial/Stock_prediction_project"

cache_dir = Path(root) / 'cache'
cache_dir.mkdir(parents=True, exist_ok=True)

from easydict import EasyDict as edict

args = edict()

args.ticker = 'BTC/USDT'
args.high_lim = 1.03
args.low_lim = 0.97
args.timeframe = '5m'

args.train_start_date = (2019,1,1,10) 
args.train_finish_date = (2022,1,1,10)

args.test_start_date = (2022,1,1,11)
args.test_finish_date = (2022,4,1,10)

args.nhid_tran = 32
args.nhead = 8
args.nlayers_transformer = 3
args.attn_pdrop = 0.1
args.resid_pdrop = 0.1
args.embd_pdrop = 0.1
args.nff = 4 * args.nhid_tran

args.x_frames = 100
args.y_frames = 5
args.batch_size = 100

args.gpu = True

args.lr_transformer = 0.0001

device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'


class StockDataset(Dataset):
    def __updown__(self, y, revenue, loss):
        open = y[0][2]
        hi = open*revenue
        lo = open*loss

        val = [0,1,0]
        for i in y:
            if i[0] > hi:
                self.freq
                val = [1,0,0]
            if i[1] < lo:
                val = [0,0,1]

        return torch.FloatTensor(val)
       
    
    def __init__(self, symbol, x_frames, y_frames, timeframe, revenue, loss, start, end):
        
        self.symbol = symbol
        self.timeframe = timeframe

        self.x_frames = x_frames
        self.y_frames = y_frames
        self.revenue = revenue
        self.loss = loss
        
        self.start = start
        self.end = end

        self.freq = []

        if os.path.exists(cache_dir / f'cache{self.start},{self.end}.csv'):
          self.data = pd.read_csv(cache_dir / f'cache{self.start},{self.end}.csv')
        else:
          self.data = Data_reader(self.symbol, self.timeframe, self.start, self.end) 
          self.data.to_csv(cache_dir / f'cache{self.start},{self.end}.csv', encoding='utf-8', index=False)
        
    def __len__(self):
        l = len(self.data) - (self.x_frames + self.y_frames) + 1
        if l < 1:
            raise NameError("too short time")
        return l
    
    def __getitem__(self, idx):
        idx += self.x_frames
        data = self.data.iloc[idx-self.x_frames:idx+self.y_frames]
        data = data[['High', 'Low', 'Open', 'Close', 'Volume']]
        #data = data.apply(lambda x: np.log(x+1) - np.log(x[self.x_frames-1]+1))
        data = data.apply(lambda x: (x+1) / (x.iloc[self.x_frames-1]+1) )
        data = data.values
        X = torch.FloatTensor(data[:self.x_frames])
        y = self.__updown__(data[self.x_frames:], self.revenue, self.loss)
        
        return X, y



train_set = StockDataset(args.ticker, args.x_frames, args.y_frames, args.timeframe, args.high_lim, args.low_lim, args.train_start_date, args.train_finish_date)
test_set = StockDataset(args.ticker, args.x_frames, args.y_frames, args.timeframe, args.high_lim, args.low_lim, args.test_start_date, args.test_finish_date)

