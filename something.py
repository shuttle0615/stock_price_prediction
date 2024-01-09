from Data.data_reader import Data_reader

pdf = Data_reader('BTC/USDT', '1h', (2020, 1, 1, 10), (2021, 1, 1, 10))

print(pdf)