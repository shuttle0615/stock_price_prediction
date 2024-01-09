import torch

# test loop
def test_loop(model, loss_fn, loader):
    running_vloss = 0.

    with torch.no_grad():
        for i, vdata in enumerate(loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    return running_vloss / (i + 1)