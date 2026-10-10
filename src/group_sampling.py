"""Bounded training-loss-based sampling with a uniform replay component."""
import torch


class DifficultySampler:
    def __init__(self, groups, device, decay=.95):
        if groups < 1 or not 0 <= decay < 1:
            raise ValueError('Invalid sampler settings')
        self.decay = decay
        self.loss_ema = torch.ones(groups, device=device)
        self.counts = torch.zeros(groups, device=device, dtype=torch.long)

    def probabilities(self):
        relative = (self.loss_ema / self.loss_ema.mean().clamp_min(1e-6)).clamp(.25, 4.)
        return .5 / len(relative) + .5 * relative / relative.sum()

    @torch.no_grad()
    def update(self, groups, per_character_loss):
        values = per_character_loss.detach().float()
        if values.shape != groups.shape or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('Expected finite nonnegative per-example losses')
        sums = torch.zeros_like(self.loss_ema).scatter_add_(0, groups, values)
        counts = torch.zeros_like(self.counts).scatter_add_(0, groups, torch.ones_like(groups))
        present = counts > 0
        means = sums / counts.clamp_min(1)
        self.loss_ema[present] = self.decay * self.loss_ema[present] + (1-self.decay) * means[present]
        self.counts += counts

    def report(self, names):
        return {name: dict(loss_ema=float(loss), probability=float(prob), draws=int(count))
                for name, loss, prob, count in zip(names, self.loss_ema, self.probabilities(), self.counts)}
