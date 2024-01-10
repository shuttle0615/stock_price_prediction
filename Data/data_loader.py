from Data import *

def Data_loader(data_args):
    
    # total data
    df, name = Data_reader(data_args.ticker, data_args.timeframe, data_args.start_date, data_args.finish_date)
    
    # process data
    new_df = Data_preprocess(df, name, data_args.x_frames, data_args.y_frames, data_args.high_lim, data_args.low_lim)
    
    # seperate train and validation
    last = len(new_df)
    idx1 = int(last * data_args.data_ratio[0])
    idx2 = int(last * data_args.data_ratio[1])
    
    train = new_df.iloc[0:idx1]
    validation = new_df.iloc[idx1:idx2]
    test = new_df.iloc[idx2:last]
    
    # create dataset
    train_set = StockDataset(train)
    validation_set = StockDataset(validation)
    test_set = StockDataset(test)
    
    # create sampler
    train_sampler = Data_sampler(train, data_args.replacement)
    validation_sampler = Data_sampler(validation, data_args.replacement)
    
    # create Dataloader 
    train_dataloader = DataLoader(train_set, batch_size=data_args.batch_size, sampler=train_sampler) 
    validation_dataloader = DataLoader(validation_set, batch_size=data_args.batch_size, sampler=validation_sampler) 
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    return train_dataloader, validation_dataloader, test_dataloader