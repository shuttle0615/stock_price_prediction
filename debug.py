from Data import Data_loader
from Model import Model_transformer
from Train import experiment
from args import data_args, model_args, train_args

tr, val = Data_loader(data_args)
print("data loader done")

mod = Model_transformer(model_args)
print("model")

ex1 = experiment(tr, val, mod, train_args)
print("exp setup done")

print("exp start")
ex1.run_exp()
print("exp end")
print(ex1.name)
