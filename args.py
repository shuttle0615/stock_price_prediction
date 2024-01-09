from easydict import EasyDict as edict

args = edict()

args.ticker = 'BTC/USDT'
args.high_lim = 1.03
args.low_lim = 0.99

args.train_start_date = (2019,1,1,10) 
args.train_finish_date = (2022,1,1,10)

args.test_start_date = (2022,1,1,11)
args.test_finish_date = (2023,1,1,10)

args.nhid_tran = 256
args.nhead = 8
args.nlayers_transformer = 6
args.attn_pdrop = 0.1
args.resid_pdrop = 0.1
args.embd_pdrop = 0.1
args.nff = 4 * args.nhid_tran

args.x_frames = 20
args.y_frames = 4
args.batch_size = 100

args.gpu = False

args.lr_transformer = 0.0001