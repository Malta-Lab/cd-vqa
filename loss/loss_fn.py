from collections import OrderedDict

from torch import nn
from torch.nn import functional as F
import torch
import inspect


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """computes log(sigmoid(logits)), log(1-sigmoid(logits))"""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def elementwise_logsumexp(a, b):
    """computes log(exp(x) + exp(b))"""
    return torch.max(a, b) + torch.log1p(torch.exp(-torch.abs(a - b)))


def renormalize_binary_logits(a, b):
    """Normalize so exp(a) + exp(b) == 1"""
    norm = elementwise_logsumexp(a, b)
    return a - norm, b - norm


class DebiasLossFn(nn.Module):
    """General API for our loss functions"""

    def forward(self, hidden, logits, bias, labels):
        """
        :param hidden: [batch, n_hidden] hidden features from the last layer in the model
        :param logits: [batch, n_answers_options] sigmoid logits for each answer option
        :param bias: [batch, n_answers_options]
          bias probabilities for each answer option between 0 and 1
        :param labels: [batch, n_answers_options]
          scores for each answer option, between 0 and 1
        :return: Scalar loss
        """
        raise NotImplementedError()

    def to_json(self):
        """Get a json representation of this loss function.

        We construct this by looking up the __init__ args
        """
        cls = self.__class__
        init = cls.__init__
        if init is object.__init__:
            return []  # No init args

        init_signature = inspect.getargspec(init)
        if init_signature.varargs is not None:
            raise NotImplementedError("varags not supported")
        if init_signature.keywords is not None:
            raise NotImplementedError("keywords not supported")
        args = [x for x in init_signature.args if x != "self"]
        out = OrderedDict()
        out["name"] = cls.__name__
        for key in args:
            out[key] = getattr(self, key)
        return out
