import ccxt
from datetime import datetime
import pandas as pd

def Data_reader(symbol, timeframe, start, end):

  binance = ccxt.binance(config={
    'options': {
        'defaultType': 'future' 
    }
  }) 

  startTime = datetime(*start)
  startTime = datetime.timestamp(startTime)
  startTime = int(startTime*1000) 

  endTime = datetime(*end)
  endTime = datetime.timestamp(endTime)
  endTime = int(endTime*1000) 

  if timeframe == '1h':
    unit_time = (60*60*1000)
  elif timeframe == '30m':
    unit_time = (30*60*1000)
  elif timeframe == '15m':
    unit_time = (15*60*1000)
  elif timeframe == '5m':
    unit_time = (5*60*1000)
  elif timeframe == '1m':
    unit_time = (60*1000)
  else:
    raise NameError("not supported time frame")

  diff = endTime - startTime
  num_unit_time = diff // unit_time

  if num_unit_time > 1000 :

    repeat = num_unit_time // 1000 
    leftover = num_unit_time % 1000 

    ohlcv = []

    for i in range(repeat):
      ohlcv = ohlcv + binance.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=(startTime + i*unit_time*1000), limit=1000)

    ohlcv = ohlcv + binance.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=(startTime + repeat*unit_time*1000), limit=leftover)    

  else: 
    ohlcv = binance.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=startTime, limit=num_unit_time)

  df = pd.DataFrame(ohlcv, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
  df['Time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['Time']]
  df.set_index('Time', inplace=True)
  
  return df

print(Data_reader('BTC/USDT', '5m', (2022,1,1,11), (2022,1,4,11)).size)