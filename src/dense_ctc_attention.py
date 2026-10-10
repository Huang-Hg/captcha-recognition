"""Controlled extensions to DFCR: learned height pooling and count supervision.

The convolution/GRU shapes stay compatible with pre-domain v2 weights.
Zero-initialized height scores exactly reproduce mean pooling at initialization.
Count predictions are auxiliary and never silently modify CTC inference.
"""
import torch
from torch import nn
from torch.nn import functional as F

from dense_ctc import CHAR_SET, DFCRv2


class DFCRv4(DFCRv2):
    def __init__(self, height_pooling='attention', count_aux=False, max_count=12,
                 horizontal_stride=2, **kwargs):
        super().__init__(**kwargs)
        if height_pooling not in ('mean', 'attention'):
            raise ValueError('height_pooling must be mean or attention')
        if horizontal_stride not in (1, 2) or max_count < 1:
            raise ValueError('Invalid stride or maximum count')
        if self.fc.out_features != len(CHAR_SET) + 1:
            raise ValueError('v4 currently uses the shared 62-character vocabulary')
        self.height_pooling = height_pooling
        self.count_aux = count_aux
        self.max_count = max_count
        self.conv1.stride = (2, horizontal_stride)
        channels = self.bn_final.num_features
        if height_pooling == 'attention':
            self.height_score = nn.Conv2d(channels, 1, 1)
            nn.init.zeros_(self.height_score.weight)
            nn.init.zeros_(self.height_score.bias)
        if count_aux:
            self.count_head = nn.Linear(channels, len(CHAR_SET) * (max_count + 1))

    def initialize_from_v2(self, state):
        # Explicit allowlist: do not silently ignore incompatible backbone keys.
        expected = {k for k in self.state_dict()
                    if k.startswith(('height_score.', 'count_head.'))}
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError(f'Incompatible v2 initialization: {result}')

    def forward(self, x, return_aux=False):
        x = self.pool1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.features(x)
        features = self.se_final(F.relu(self.bn_final(x), inplace=True))
        weights = None
        if self.height_pooling == 'attention':
            weights = self.height_score(features).float().softmax(dim=2).to(features.dtype)
            sequence = (features * weights).sum(dim=2)
        else:
            sequence = features.mean(dim=2)
        sequence, _ = self.rnn(sequence.permute(0, 2, 1))
        logits = self.fc(sequence)
        if not return_aux:
            return logits
        counts = None
        if self.count_aux:
            counts = self.count_head(features.mean(dim=(2, 3))).reshape(
                x.shape[0], len(CHAR_SET), self.max_count + 1)
        return dict(logits=logits, count_logits=counts, height_weights=weights)


def character_count_targets(targets, lengths, max_count=12):
    """Build exact counts from flattened CTC targets, including mixed lengths."""
    lengths = lengths.to(device=targets.device, dtype=torch.long)
    if targets.ndim != 1 or lengths.ndim != 1 or lengths.numel() == 0:
        raise ValueError('Expected nonempty lengths and flattened targets')
    if (lengths < 1).any() or int(lengths.sum()) != targets.numel():
        raise ValueError('Invalid target lengths')
    if ((targets < 0) | (targets >= len(CHAR_SET))).any():
        raise ValueError('CTC blank and unknown characters are not count targets')
    rows = torch.repeat_interleave(torch.arange(len(lengths), device=targets.device), lengths)
    counts = torch.zeros(len(lengths) * len(CHAR_SET), device=targets.device, dtype=torch.long)
    counts.scatter_add_(0, rows * len(CHAR_SET) + targets, torch.ones_like(targets))
    counts = counts.reshape(len(lengths), len(CHAR_SET))
    if (counts > max_count).any():
        raise ValueError('Character count exceeds configured max_count')
    return counts


def character_count_loss(logits, counts):
    """Balance present/absent characters so the many zeros cannot dominate."""
    loss = F.cross_entropy(logits.float().transpose(1, 2), counts, reduction='none')
    present = counts > 0
    terms = []
    for mask in (present, ~present):
        if mask.any():
            terms.append(loss[mask].mean())
    return torch.stack(terms).mean()
