import errno
import os
import numpy as np
import torch
import torch.nn as nn

from loss.learned_mixin import LearnedMixin

EPS = 1e-7


def select_loss_fn(model, args):
    model.debias_loss_fn = LearnedMixin(
        args.entropy_penalty, debias_w=args.debias_weight)


def get_num_ftrs(model):
    if isinstance(model.classifier, nn.Sequential):
        return model.classifier[0].in_features
    return model.classifier.in_features


def add_final_layer(model, n_labels):
    num_ftrs = get_num_ftrs(model)
    model.classifier = nn.Linear(num_ftrs, n_labels)
    return model


def load_pretrained_model(model_path, model, device, n_labels=None):
    print(f'Loading model from {model_path}....')
    weights = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(torch.load(weights))
    except:
        state = model.state_dict()
        for name, param in model.named_parameters():
            if name in weights.keys():
                state[name] = weights[name]
        model.load_state_dict(state)
    model.to(device)
    if n_labels:
        model = add_final_layer(model, n_labels)
    print(f'Model loaded from {model_path}.')
    return model


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)
