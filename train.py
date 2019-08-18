#!/usr/bin/env python
# coding: utf-8

# In[89]:


from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import ExpressionClassifier

# In[60]:


train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
topil = transforms.ToPILImage()
totensor = transforms.Compose(valid_transforms.transforms[:-1])

# In[87]:


train_set = ImageFolder(root='/home/khairulimam/datasets/expressions/IMFDB/train/', transform=train_transforms)
valid_set = ImageFolder(root='/home/khairulimam/datasets/expressions/IMFDB/valid/', transform=valid_transforms)
test_set = list(Path('/home/khairulimam/datasets/lfw-deepfunneled/').glob('*/*.jpg'))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True)

# In[83]:


classes = valid_set.classes


# In[103]:


def predict(imgpath):
    img = Image.open(imgpath)
    x = valid_transforms(img)

    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        _, p = torch.max(logits, 1)

    plt.text(115, 10, classes[p], fontweight='bold', horizontalalignment='center',
             bbox=dict(facecolor='white'))
    plt.imshow(totensor(img).permute(1, 2, 0))
    plt.axis('off')
    plt.show()


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ExpressionClassifier(num_classes=7)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.to(device)
model = torch.nn.DataParallel(model)
criterion = torch.nn.CrossEntropyLoss()


# In[34]:


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


for epoch in range(10):
    lossses = list()
    accuracies = list()
    for idx, (imgs, lbls) in enumerate(train_loader):
        loss = train(model, imgs, lbls)
        lossses.append(loss)
    l = sum(lossses) / len(lossses)
    print(epoch, 'train loss', l)
    for idx, (imgs, lbls) in enumerate(valid_loader):
        accuracy = validate(model, imgs, lbls)
        accuracies.append(accuracy)
    ac = sum(accuracies) / len(accuracies)
    print(epoch, 'valid accuracies', ac)
    torch.save(dict(
        state=model.cpu().module.state_dict(),
        optimizer=optimizer.state_dict()
    ), 'model.pth')
    write(f'{l},{ac}\n')