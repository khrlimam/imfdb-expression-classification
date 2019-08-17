from pathlib import Path

import torch


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
