from Train import *

def train_loop(model, optimizer, loss_fn, loader):
    running_loss = 0.
    last_loss = 0.
    l = len(loader)

    for i, data in enumerate(loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs.to(device))

        loss = loss_fn(outputs, labels.to(device))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch - what is the number of batch??
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss
    


    
