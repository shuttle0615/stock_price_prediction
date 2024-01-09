
from Data import *

class StockDataset(Dataset):
    def __vectorize__(self, r):
      r = int(r)
      if r == 0:
        return [1,0,0]
      elif r == 1:
        return [0,1,0]
      elif r == 2:
        return [0,0,1]

    def __init__(self, df):
      self.data = df
      self.all_X = [eval(i) for i in self.data["X"].values.tolist()]
      self.all_y = self.data["label"].apply(self.__vectorize__)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.all_X[idx])
        y = torch.FloatTensor(self.all_y[idx])

        return X, y

