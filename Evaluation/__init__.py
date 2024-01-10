import torch
import os
from pathlib import Path
from Data import root
from sklearn.metrics import confusion_matrix

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

result_dir = Path(root) / 'results'
result_dir.mkdir(parents=True, exist_ok=True)

def predictor(exp, model, test) :
    #load trained model
    test_model = model
    if os.path.exists(result_dir / exp.name):
        test_model.load_state_dict(torch.load(result_dir / exp.name))
        
    #test data
    model.eval()

    l_voutput = []
    l_vlabel = []
    with torch.no_grad():
        for i, vdata in enumerate(test):
            vinput, vlabel = vdata
            voutput = model(vinput.to(device))
            if voutput[0][0] > 0.4:
                voutput = 0
            elif voutput[0][2] > 0.4:
                voutput = 2
            else:
                voutput = 1

            l_voutput.append(voutput)
            l_vlabel.append(vlabel[0].reshape((3,)))
            print(vlabel[0])
            break

    return l_voutput, l_vlabel
        