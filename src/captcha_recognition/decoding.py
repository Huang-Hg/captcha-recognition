"""Sequence-level CTC scoring, including every alignment path and view."""
import math
import torch


def ctc_mixture_scores(logits, candidates, blank):
    if logits.ndim != 3 or not candidates:
        raise ValueError('Require view/time/class logits and at least one candidate')
    views, steps, _ = logits.shape
    probabilities = logits.float().log_softmax(-1)
    repeated = probabilities.repeat(len(candidates), 1, 1).transpose(0, 1)
    lengths = torch.tensor([len(c) for c in candidates for _ in range(views)], dtype=torch.long)
    targets = torch.tensor([char for c in candidates for _ in range(views) for char in c],
                           device=logits.device, dtype=torch.long)
    losses = torch.nn.functional.ctc_loss(repeated, targets, torch.full_like(lengths, steps), lengths,
        blank=blank, reduction='none', zero_infinity=False)
    return torch.logsumexp(-losses.reshape(len(candidates), views), dim=1) - math.log(views)
