"""Custom architecture hypotheses, not claimed reproductions or novel publications."""
import torch
from torch import nn
import torch.nn.functional as F
from .config import HybridConfig
from .model import HybridRecognizer, load_hybrid
from svtr_adapter import UPSTREAM_COMMIT


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class PixelDetail(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU())
        self.project = zero_output(nn.Conv2d(96, channels, 1))

    def forward(self, features, images, widths):
        detail = F.adaptive_avg_pool2d(self.stem(images), features.shape[-2:])
        return features+self.project(detail)


class ColumnRecurrent(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.height_scores = nn.Conv2d(channels, 1, 1)
        self.norm = nn.LayerNorm(channels)
        self.context = nn.GRU(channels, 128, batch_first=True, bidirectional=True)
        self.project = zero_output(nn.Linear(256, channels))

    def forward(self, features, images, widths):
        weights = self.height_scores(features).softmax(2)
        columns = (features*weights).sum(2).transpose(1, 2)
        lengths = torch.full((len(features),), columns.shape[1], dtype=torch.long)
        if widths is not None:
            lengths = torch.ceil(widths.float()*columns.shape[1]/images.shape[-1]).long().clamp(1, columns.shape[1]).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.norm(columns), lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.context(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=columns.shape[1])
        return features+self.project(output).transpose(1, 2)[:, :, None]


class SpatialExperts(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(channels, 96, 1), nn.GroupNorm(8, 96), nn.GELU())
        self.local = nn.Conv2d(96, 96, 3, padding=1, groups=96)
        self.wide = nn.Conv2d(96, 96, (3, 9), padding=(1, 4), groups=96)
        self.global_project = nn.Conv2d(96, 96, 1)
        self.router = nn.Conv2d(96, 3, 1)
        self.project = zero_output(nn.Conv2d(96, channels, 1))

    def forward(self, features, images, widths):
        reduced = self.reduce(features)
        valid = torch.ones(len(features), 1, 1, features.shape[-1], device=features.device)
        if widths is not None:
            lengths = torch.ceil(widths.float()*features.shape[-1]/images.shape[-1]).clamp(1, features.shape[-1])
            valid = (torch.arange(features.shape[-1], device=features.device)[None, None, None] < lengths[:, None, None, None]).to(features.dtype)
        context = (reduced*valid).sum((2, 3), keepdim=True)/(valid.sum((2, 3), keepdim=True)*features.shape[2])
        experts = torch.stack((self.local(reduced), self.wide(reduced), self.global_project(context).expand_as(reduced)), dim=1)
        routing = self.router(reduced).softmax(1)[:, :, None]
        return features+self.project((experts*routing).sum(1))


class ColumnRectifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.offset = nn.Sequential(nn.Conv1d(channels, 96, 3, padding=1), nn.GELU(), zero_output(nn.Conv1d(96, 2, 1)))

    def forward(self, features, images, widths):
        height, width = features.shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=features.device),
                              torch.linspace(-1, 1, width, device=features.device), indexing='ij')
        identity = torch.stack((x, y), -1)[None].expand(len(features), -1, -1, -1)
        offsets = self.offset(features.mean(2)).float().tanh().transpose(1, 2)[:, None]
        scale = features.new_tensor([.06, .3], dtype=torch.float32)
        # Subtract an identically sampled reference: zero offsets are an exact
        # identity even when interpolation would round the original grid.
        warped = F.grid_sample(features.float(), identity+scale*offsets, align_corners=True, padding_mode='border')
        reference = F.grid_sample(features.float(), identity, align_corners=True, padding_mode='border')
        return features+(warped-reference).to(features.dtype)


VARIANTS = dict(pixel_detail=PixelDetail, column_recurrent=ColumnRecurrent,
                spatial_experts=SpatialExperts, column_rectifier=ColumnRectifier)


class ExperimentalRecognizer(HybridRecognizer):
    architecture = 'dfcr_experimental_v1'

    def __init__(self, visual_config, config, variant):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown experimental architecture: {variant}')
        super().__init__(visual_config, config)
        self.variant = variant
        self.variant_block = VARIANTS[variant](self.encoder.out_channels)

    @classmethod
    def from_parent(cls, checkpoint, config, variant):
        if checkpoint.get('architecture') == cls.architecture:
            model = load_experimental(checkpoint)
            if model.config != config or model.variant != variant:
                raise ValueError('Continuing an experiment requires its recorded variant and geometry')
            return model
        parent = load_hybrid(checkpoint)
        if parent.config != config:
            raise ValueError('Architecture experiments require identical parent geometry/vocabulary')
        model = cls(checkpoint['model_config'], config, variant)
        mismatch = model.load_state_dict(parent.state_dict(), strict=False)
        if mismatch.unexpected_keys or any(not key.startswith('variant_block.') for key in mismatch.missing_keys):
            raise ValueError('Unexpected parent transfer mismatch')
        return model

    def adapt_features(self, features, images, valid_widths):
        return self.variant_block(features, images, valid_widths)

    def forward(self, images):
        # SVTRInput's normalized zero padding is distinct from uint8 source
        # pixels. Preserve width-aware branches for legacy logit-only callers.
        content = images.ne(0).any(1).any(1)
        positions = torch.arange(1, images.shape[-1]+1, device=images.device)
        widths = (content*positions).amax(1).clamp_min(1)
        return self.encode(images, widths)['ctc']

    def checkpoint(self, **metadata):
        return super().checkpoint(experimental_variant=self.variant, **metadata)


def load_experimental(checkpoint):
    if checkpoint.get('architecture') != ExperimentalRecognizer.architecture or checkpoint.get('schema_version') != 2:
        raise ValueError('Invalid experimental schema')
    if checkpoint.get('upstream_commit') != UPSTREAM_COMMIT or checkpoint.get('preprocessing') != 'svtr_bgr_aspect_pad':
        raise ValueError('Invalid experimental source/preprocessing')
    config = HybridConfig(**checkpoint['hybrid_config'])
    if config.alphabet != checkpoint['alphabet'] or checkpoint['input_size'] != [config.height, config.width]:
        raise ValueError('Inconsistent experimental metadata')
    model = ExperimentalRecognizer(checkpoint['model_config'], config, checkpoint['experimental_variant'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model
