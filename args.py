from easydict import EasyDict as edict

data_args = edict()

data_args.ticker = 'BTC/USDT'# dataset
data_args.timeframe = '1h'#dataset
data_args.start_date = (2020,10,1,10) #dataset
data_args.finish_date = (2022,10,1,10) #dataset

data_args.x_frames = 100 #dataset
data_args.y_frames = 5 #dataset

data_args.high_lim = 1.03 #datset
data_args.low_lim = 0.97 #dataset

data_args.train_vs_validation_ratio = 0.7
data_args.batch_size = 100 #dataset
data_args.replacement = True

args = edict()

args.nhid_tran = 32 #model
args.nhead = 8 #model
args.nlayers_transformer = 3 #model
args.attn_pdrop = 0.1 #model
args.resid_pdrop = 0.1 #model
args.embd_pdrop = 0.1 #model
args.nff = 4 * args.nhid_tran #model

args.epoch = 5

args.gpu = True

args.lr_transformer = 0.0001