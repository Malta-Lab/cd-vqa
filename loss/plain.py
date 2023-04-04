from torch.nn import functional as F

from loss.loss_fn import DebiasLossFn


class Plain(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss *= labels.size(1)
        return loss
