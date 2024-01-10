from easydict import EasyDict as edict

data_args = edict()

data_args.ticker = 'BTC/USDT'# dataset
data_args.timeframe = '1h'#dataset
data_args.start_date = (2020,10,1,10) #dataset
data_args.finish_date = (2023,10,1,10) #dataset

data_args.x_frames = 100 #dataset
data_args.y_frames = 5 #dataset

data_args.high_lim = 1.03 #datset
data_args.low_lim = 0.97 #dataset

data_args.train_vs_validation_ratio = 0.7
data_args.batch_size = 100 #dataset
data_args.replacement = True

model_args = edict()

model_args.nhid_tran = 32 #model
model_args.nhead = 8 #model
model_args.nlayers_transformer = 3 #model
model_args.attn_pdrop = 0.1 #model
model_args.resid_pdrop = 0.1 #model
model_args.embd_pdrop = 0.1 #model
model_args.nff = 4 * model_args.nhid_tran #model

train_args = edict()

train_args.epoch = 20
train_args.lr = 0.0001