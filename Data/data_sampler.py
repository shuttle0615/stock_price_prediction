from Data import *

def Data_sampler(df, replacement):
  class_sample_count = [(df["label"] == 0).sum(), (df["label"] == 1).sum(), (df["label"] == 2).sum()]
  weight = 1. / torch.tensor(class_sample_count, dtype=torch.float)

  samples_weights = weight[df["label"].values.tolist()]
  samples_weights = samples_weights.to(device)

  return WeightedRandomSampler(samples_weights, len(samples_weights), replacement=replacement)