from Train import *

def validation_loop(model, loss_fn, loader):
    running_vloss = 0.

    with torch.no_grad():
        for i, vdata in enumerate(loader):
            vinputs, vlabels = vdata

            voutputs = model(vinputs.to(device))

            vloss = loss_fn(voutputs, vlabels.to(device))
            
            running_vloss += vloss

    return running_vloss / (i + 1)