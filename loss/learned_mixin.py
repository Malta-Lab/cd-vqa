import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from loss.loss_fn import DebiasLossFn, convert_sigmoid_logits_to_binary_logprobs, renormalize_binary_logits, elementwise_logsumexp


class LearnedMixin(DebiasLossFn):
    def __init__(self, w, debias_w=1, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        :param w: Weight of the entropy penalty
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixin, self).__init__()
        self.w = w
        self.smooth_init = smooth_init
        self.constant_smooth = constant_smooth
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.smooth = smooth
        self.debias_w = debias_w
        if self.smooth:
            self.smooth_param = torch.nn.Parameter(
              torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
        else:
            self.smooth_param = None

    def forward(self, hidden, logits, bias, labels):
        factor = self.bias_lin.forward(hidden)  # [batch, 1]
        factor = F.softplus(factor)

        bias = torch.stack([bias, 1 - bias], 2)  # [batch, n_answers, 2]

        # Smooth
        bias += self.constant_smooth
        if self.smooth:
            soften_factor = F.sigmoid(self.smooth_param)
            bias = bias + soften_factor.unsqueeze(1)

        bias = torch.log(bias)  # Convert to logspace

        # Scale by the factor
        # [batch, n_answers, 2] * [batch, 1, 1] -> [batch, n_answers, 2]
        bias = bias * factor.unsqueeze(1)

        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
        log_probs = torch.stack([log_prob, log_one_minus_prob], 2)

        # Add the bias in
        logits = bias + log_probs

        # Renormalize to get log probabilities
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        # Compute loss
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0)

        # Re-normalized version of the bias
        bias_norm = elementwise_logsumexp(bias[:, :, 0], bias[:, :, 1])
        bias_logprob = bias - bias_norm.unsqueeze(2)

        # Compute and add the entropy penalty
        entropy = -(torch.exp(bias_logprob) * bias_logprob).sum(2).mean()
        return self.debias_w*(loss + self.w * entropy)
