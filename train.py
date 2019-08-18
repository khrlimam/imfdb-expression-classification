import argparse
import time

import torch

from dataloader import get_train_valid_data
from model import ExpressionClassifier


def train(model, imgs, lbls):
    model.train()
    imgs = imgs.to(device)
    lbls = lbls.to(device)

    logits = model(imgs)
    loss = criterion(logits, lbls)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def validate(model, imgs, lbls):
    model.eval()
    with torch.no_grad():
        imgs = imgs.to(device)
        lbls = lbls.to(device)

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
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExpressionClassifier(num_classes=7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    try:
        saved_state = torch.load('model.pth')
        model_state = saved_state['state']
        optim_state = saved_state['optim']

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)
        print('model loaded!')
    except Exception:
        pass

    model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader, valid_loader = get_train_valid_data(batch_size)

    print("start training")
    start = time.time()
    print('epoch', '\t', 'Train loss', '\t', 'Valid accuracy')
    for epoch in range(epochs):
        lossses = list()
        accuracies = list()
        for idx, (imgs, lbls) in enumerate(train_loader):
            loss = train(model, imgs, lbls)
            lossses.append(loss)
        l = sum(lossses) / len(lossses)
        for idx, (imgs, lbls) in enumerate(valid_loader):
            accuracy = validate(model, imgs, lbls)
            accuracies.append(accuracy)
        ac = (sum(accuracies) / len(accuracies)) / batch_size
        print(epoch, '\t', l, '\t', ac)
        torch.save(dict(
            state=model.module.state_dict(),
            optim=optimizer.state_dict()
        ), 'model.pth')
        write(f'{l},{ac}\n')
    print("Training finished on ", time.time() - start, 'seconds')
