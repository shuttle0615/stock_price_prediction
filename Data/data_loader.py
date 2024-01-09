from Data import *

def Data_loader(data_args):
    
    # total data
    df, name = Data_reader(data_args.ticker, data_args.timeframe, data_args.start_date, data_args.finish_date)
    
    # process data
    new_df = Data_preprocess(df, name, data_args.x_frames, data_args.y_frames, data_args.high_lim, data_args.low_lim)
    
    # seperate train and validation
    idx = int(len(new_df) * data_args.train_vs_validation_ratio)
    
    train = new_df[:idx]
    validation = new_df[idx:]
    
    # create dataset
    train_set = StockDataset(train)
    validation_set = StockDataset(validation)
    
    # create sampler
    train_sampler = Data_sampler(train, data_args.replacement)
    validation_sampler = Data_sampler(validation, data_args.replacement)
    
    # create Dataloader 
    train_dataloader = DataLoader(train_set, batch_size=data_args.batch_size, sampler=train_sampler) 
    test_dataloader = DataLoader(validation_set, batch_size=data_args.batch_size, sampler=validation_sampler) 
    
    return train_dataloader, test_dataloader