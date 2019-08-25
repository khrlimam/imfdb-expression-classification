import argparse
import time

import torch

from dataloader import get_train_valid_data
from model import ExpressionMobileNet, ExpressionSqueezeNet


def train(model, optimizer, imgs, lbls):
    model.train()
    logits = model(imgs)
    loss = criterion(logits, lbls)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def validate(model, imgs, lbls):
    model.eval()
    with torch.no_grad():
        logits = model(imgs)

        _, predictions = torch.max(logits, 1)

        trues = predictions == lbls
        return trues.sum().item()


# In[36]:

def write(data):
    with open('log.csv', 'a') as f:
        f.write(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=64, type=int, help='split data into number of batches')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='use given batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='use given momentum')
    parser.add_argument('--model', default='mobilenet', type=str,
                        help='Choose which model to load for training. 1. squeezenet, 2.mobilenet')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    which_model = args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExpressionMobileNet() if which_model == 'mobilenet' else ExpressionSqueezeNet()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    try:
        saved_state = torch.load(f'{which_model}.pth')
        model_state = saved_state['state']
        model.load_state_dict(model_state)
        print('model loaded!')
    except Exception:
        print('failed load model')

    try:
        optim_state = saved_state['optim']
        optimizer.load_state_dict(optim_state)
    except Exception:
        print('failed load optimizer state')

    model = torch.nn.DataParallel(model)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader, valid_loader = get_train_valid_data(batch_size)

    print("start training")
    start = time.time()
    print('epoch', '\t', 'Train loss', '\t', 'Valid accuracy')
    for epoch in range(epochs):
        lossses = list()
        accuracies = list()
        for idx, (imgs, lbls) in enumerate(train_loader):
            loss = train(model, optimizer, imgs.to(device), lbls.to(device))
            lossses.append(loss)
        l = sum(lossses) / len(lossses)
        for idx, (imgs, lbls) in enumerate(valid_loader):
            accuracy = validate(model, imgs.to(device), lbls.to(device))
            accuracies.append(accuracy)
        ac = (sum(accuracies) / len(accuracies)) / batch_size
        print(epoch, '\t', round(l, 5), '\t', round(ac, 5))
        torch.save(dict(
            state=model.module.state_dict(),
            optim=optimizer.state_dict()
        ), f'{which_model}.pth')
        write(f'{l},{ac}\n')
    print("Training finished on ", time.time() - start, 'seconds')
