from Data import *

def preprocess(df, name, x_frames, y_frames, revenue, loss):

  #check cache
  if os.path.exists(cache_dir / (name + '_processed.csv')):
    return pd.read_csv(cache_dir / (name + '_processed.csv'))

  #helper function
  def updown(y, revenue, loss):
    open = y[0][2]
    hi = open*revenue
    lo = open*loss

    val = 1
    for i in y:
        if i[0] > hi:
            val = 0
        if i[1] < lo:
            val = 2

    return val

  #initialize the data
  X = []
  y = []
  label = []

  #length of data
  l = len(df) - (x_frames + y_frames) + 1
  if l < 1:
    raise NameError("x and y frames are too big")

  #process data
  for idx in range(x_frames, x_frames + l):
    data = df.iloc[idx-x_frames:idx+y_frames]
    data = data[['High', 'Low', 'Open', 'Close', 'Volume']]
    data = data.apply(lambda x: (x+1) / (x.iloc[x_frames-1]+1) )
    data = data.values
    X.append(data[:x_frames].tolist())
    y.append(data[x_frames:].tolist())
    label.append(updown(data[x_frames:], revenue, loss))


  #save data
  new_df = pd.DataFrame({"X":X, 'y':y, 'label':label})
  new_df.to_csv(cache_dir / (name + '_processed.csv'), encoding='utf-8', index=False)

  return new_df