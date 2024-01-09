from Train import train_loop
from Test import test_loop
from datetime import datetime
from args import args

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Model.Model import TransformerEncoder
from Data.data_set import StockDataset


# data loader
train_set = StockDataset(args.ticker, args.x_frames, args.y_frames, args.high_lim, args.low_lim, args.train_start_date, args.train_finish_date)
test_set = StockDataset(args.ticker, args.x_frames, args.y_frames, args.high_lim, args.low_lim, args.test_start_date, args.test_finish_date)

train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
print("data loader done")

# model
model = TransformerEncoder()

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), args.lr_transformer)

# timer
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
print("start: {}".format(timestamp))

# epoch setting
EPOCHS = 5

# v_loss setting
best_vloss = 1_000_000

# run epoch
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch))

    model.train()
    avg_loss = train_loop(model, optimizer, loss_fn, train_dataloader)

    avg_v_loss = test_loop(model, loss_fn, train_dataloader)

    print('LOSS train {} valid {}'.format(avg_loss, avg_v_loss))

    '''
    if avg_v_loss < best_vloss:
        best_vloss = avg_v_loss
        model_path = 'model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)
    '''
# finish timer
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    