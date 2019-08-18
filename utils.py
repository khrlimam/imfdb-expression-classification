from pathlib import Path

import torch
from PIL import Image
from matplotlib import pyplot as plt

from dataloader import valid_transforms, totensor


class ModelSaver():

    def __init__(self):
        self._previous_acc = 0.
        self._current_acc = 0.

    @property
    def previous_acc(self):
        return self._previous_acc

    @property
    def current_acc(self):
        return self._current_acc

    @current_acc.setter
    def current_acc(self, value):
        self._current_acc = value

    @previous_acc.setter
    def previous_acc(self, value):
        self._previous_acc = value

    def __set_accuracy(self, accuracy):
        self.previous_acc, self.current_acc = self.current_acc, accuracy

    def save_if_best(self, accuracy, state):
        if accuracy > self.current_acc:
            self.__set_accuracy(accuracy)
            torch.save(state, 'log/best_state.pth')


def create_if_not_exist(path):
    path = Path(path)
    if not path.exists(): path.touch()


def init_log_just_created(path):
    create_if_not_exist(path)
    with open(path, 'r') as f:
        if len(f.readlines()) <= 0:
            init_log_line(path)


def init_log_line(path):
    with open(path, 'w') as f:
        f.write('time,epoch,acc,loss,layers,bs,lr\n')


def predict(model, imgpath, classes):
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
