from Data import *
from Model import Model_transformer
from Train import experiment
from args import data_args, model_args, train_args

from Evaluation import *

a, b, c = Data_loader(data_args)

for out, lab in c:
    print(lab.argmax().item())
    print(type(lab))
    break


