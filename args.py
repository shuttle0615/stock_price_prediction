from easydict import EasyDict as edict

data_args = edict()

data_args.ticker = 'BTC/USDT'# dataset
data_args.timeframe = '1h'#dataset
data_args.start_date = (2020,10,1,10) #dataset
data_args.finish_date = (2023,10,1,10) #dataset

data_args.x_frames = 50 #dataset
data_args.y_frames = 1 #dataset

data_args.high_lim = 1.015 #datset
data_args.low_lim = 0.985 #dataset

data_args.data_ratio = [0.7, 0.9]
data_args.batch_size = 200 #dataset
data_args.replacement = True

model_args = edict()

model_args.nhid_tran = 32 #model
model_args.nhead = 8 #model
model_args.nlayers_transformer = 3 #model
model_args.attn_pdrop = 0.1 #model
model_args.resid_pdrop = 0.1 #model
model_args.embd_pdrop = 0.1 #model
model_args.nff = 4 * model_args.nhid_tran #model

train_args = []
lr = [0.0007, 0.0006, 0.0005, 0.0004, 0.0003]
#[0.001, 0.0005, 0.0001, 0.00005, 0.00001]


for i in lr:
    temp = edict()
    temp.epoch = 10
    temp.lr = i
    train_args.append(temp)