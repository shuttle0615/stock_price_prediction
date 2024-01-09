import torch
# train loop
def train_loop(model, optimizer, loss_fn, loader):
    running_loss = 0.
    last_loss = 0.
    l = len(loader)
    
    for i, data in enumerate(loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs.float())

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch - what is the number of batch??
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    
    return last_loss
    


    
