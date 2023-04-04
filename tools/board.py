from collections import defaultdict
import os
import shutil

import numpy
import torchvision
from torch.utils.tensorboard import SummaryWriter


class BoardAlreadyExists(Exception):
    def __init__(self, name):
        super().__init__(f'Tensorboard: {name} already exists')


class Board():
    def __init__(self, name=None, path=None, delete=False):
        path = './runs/' if path is None else path

        if os.path.exists(path) is True and delete is True:
            shutil.rmtree(f'{path}')
        elif os.path.exists(path) is True and delete is False:
            raise BoardAlreadyExists(path)

        if os.path.exists(path) is False:
            os.makedirs(path)

        if name is None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(f'{path}')

        self.epoch = defaultdict(int)
        self.histograms = defaultdict(list)

    def advance(self, epoch=None):
        if self.histograms:
            self.__save_histogram()
            self.histograms = defaultdict(list)

        if epoch is not None:
            self.epoch[epoch] += 1
        elif len(self.epoch.keys()) > 0:
            for key in self.epoch:
                self.epoch[key] += 1
        else:
            self.epoch['default'] += 1

    def close(self):
        self.writer.flush()
        self.writer.close()

    def add_grid(self, train, epoch=None, **kwargs):
        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        for i in kwargs:
            grid = torchvision.utils.make_grid(kwargs[i], normalize=True)
            if train:
                self.writer.add_image(f'Train/{i}', grid, epoch)
            else:
                self.writer.add_image(f'Validation/{i}', grid, epoch)

    def add_image(self, title, value, epoch=None):
        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        self.writer.add_image(title, value, epoch)

    def add_scalars(self, prior, epoch=None, **kwargs):
        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        for i in kwargs:
            self.writer.add_scalar(f'{prior}/{i}', kwargs[i], epoch)

    def add_histogram(self, title, histogram):
        if histogram.is_cuda:
            histogram = histogram.cpu().tolist()
        else:
            histogram = histogram.tolist()

        self.histograms[title].extend(histogram)

    def add_hparams(self, params):
        self.writer.add_hparams(params, {})

    def add_graph(self, model, data):
        self.writer.add_graph(model, data)

    def __save_histogram(self):
        for title in self.histograms:
            self.writer.add_histogram(title, numpy.array(self.histograms[title]), self.epoch['default'])

    def add_scalar(self, title, value, epoch=None):
        epoch = self.epoch['default'] if epoch is None else self.epoch[epoch]

        self.writer.add_scalar(title, value, epoch)
