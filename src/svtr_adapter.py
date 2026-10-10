"""Case-sensitive SVTRv2 transfer adapter using a pinned OpenOCR checkout.

The upstream CTC decoder detaches its input when ``use_guide`` is enabled
for joint training. A standalone CTC task must disable that detachment.
No downloads, cache writes, or implicit checkpoint loading occur here.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dense_ctc import CHAR_SET, BLANK

UPSTREAM_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
PRETRAINED_SHA256 = '3be361b72dc52e8d4f5569ae822d263b4e70671486bbaa45605711d17b8aec27'


class SVTRInput:
    """Match upstream RecDynamicResize: BGR, height 48, zero-normalized pad.

    Current task images fit width 320. Wider inputs explicitly fail, rather
    than silently cropping characters or changing the checkpoint transform.
    """
    def __init__(self, height=48, width=320):
        self.height, self.width = height, width

    def __call__(self, image):
        import cv2
        image = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
        width = math.ceil(self.height * image.shape[1] / image.shape[0])
        if width > self.width:
            raise ValueError(f'Input aspect ratio needs width {width}, exceeds {self.width}')
        resized = cv2.resize(image, (width, self.height)).astype(np.float32)
        result = np.zeros((3, self.height, self.width), dtype=np.float32)
        result[:, :, :width] = resized.transpose(2, 0, 1) / 127.5 - 1
        return torch.from_numpy(result)


class SVTRCaptcha(nn.Module):
    def __init__(self, architecture, upstream):
        super().__init__()
        upstream = Path(upstream).resolve()
        if not (upstream / 'openrec/modeling/__init__.py').is_file():
            raise FileNotFoundError(f'OpenOCR source checkout missing: {upstream}')
        sys.path.insert(0, str(upstream))
        from openrec.modeling import build_model
        self.architecture_config = copy.deepcopy(architecture)
        self.network = build_model(copy.deepcopy(architecture))
        self.network.decoder.svtr_encoder.use_guide = False

    def sequence_features(self, images):
        features = self.network.encoder(images)
        features = self.network.decoder.svtr_encoder(features)
        return features.flatten(2).transpose(1, 2)

    def forward(self, images):
        features = self.sequence_features(images)
        # Upstream forward softmaxes at eval time; our API always returns logits.
        return self.network.decoder.fc(features)

    @classmethod
    def from_pretrained(cls, upstream, weights):
        import hashlib
        import yaml
        upstream = Path(upstream)
        if hashlib.sha256(Path(weights).read_bytes()).hexdigest() != PRETRAINED_SHA256:
            raise ValueError('Pretrained checkpoint hash mismatch')
        chars = ['blank'] + (upstream / 'tools/utils/ppocr_keys_v1.txt').read_text().splitlines() + [' ']
        config = yaml.safe_load((upstream / 'configs/rec/svtrv2/svtrv2_ch.yml').read_text())['Architecture']
        config['Decoder']['out_channels'] = len(chars)
        model = cls(config, upstream)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        # Released training weights also contain the documented auxiliary GTC
        # branch. It is absent from the official CTC inference architecture.
        # Drop only these named branches, then strictly match every CTC tensor.
        state = {k: v for k, v in state.items()
                 if not k.startswith(('decoder.before_gtc.', 'decoder.gtc_head.'))}
        model.network.load_state_dict(state, strict=True)
        old = model.network.decoder.fc
        selected = [chars.index(c) for c in CHAR_SET] + [0]
        head = nn.Linear(old.in_features, len(CHAR_SET) + 1)
        with torch.no_grad():
            head.weight.copy_(old.weight[selected])
            head.bias.copy_(old.bias[selected])
        model.network.decoder.fc = head
        model.architecture_config['Decoder']['out_channels'] = BLANK + 1
        return model

    def freeze_batch_statistics(self):
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()


def alphabet_type(label):
    """Disjoint targets derived only from the supplied transcription."""
    lower = any('a' <= c <= 'z' for c in label)
    upper = any('A' <= c <= 'Z' for c in label)
    return 3 if lower and upper else 1 if lower else 2 if upper else 0


class SVTRTypedCaptcha(SVTRCaptcha):
    """Image-conditioned alphabet prior; never uses a website or target hint.

    Digits and blank remain possible in every type. A probability floor keeps
    uncertain type decisions from deleting uppercase/lowercase hypotheses.
    Centering against the initial prior preserves base logits at conversion.
    """
    def __init__(self, architecture, upstream):
        super().__init__(architecture, upstream)
        width = self.network.decoder.fc.in_features
        self.type_head = nn.Linear(width, 4)
        nn.init.zeros_(self.type_head.weight)
        with torch.no_grad():
            self.type_head.bias.copy_(torch.tensor([0., 0., 0., 4.]))
        allowed = torch.ones(4, BLANK + 1)
        for index, char in enumerate(CHAR_SET):
            if char.islower():
                allowed[:, index] = torch.tensor([0., 1., 0., 1.])
            elif char.isupper():
                allowed[:, index] = torch.tensor([0., 0., 1., 1.])
        self.register_buffer('type_allowed', allowed)
        self.register_buffer('initial_log_prior',
            (self.type_head.bias.detach().softmax(-1) @ allowed).clamp_min(.1).log())

    @classmethod
    def from_base(cls, base, upstream):
        model = cls(base.architecture_config, upstream)
        model.network.load_state_dict(base.network.state_dict(), strict=True)
        return model

    def forward_with_type(self, images):
        features = self.sequence_features(images)
        type_logits = self.type_head(features.mean(1))
        # Explicitly disable autocast for this small matrix product: otherwise
        # BF16 rounds the centered prior and changes base logits at conversion.
        with torch.autocast(device_type=type_logits.device.type, enabled=False):
            prior = (type_logits.float().softmax(-1) @ self.type_allowed.float()).clamp_min(.1)
        logits = self.network.decoder.fc(features)
        return logits.float() + (prior.log() - self.initial_log_prior)[:, None, :], type_logits

    def forward(self, images):
        return self.forward_with_type(images)[0]


class SVTRDetailCaptcha(SVTRCaptcha):
    """Learn height aggregation and retain width/4 CTC features (80 steps).

    The pretrained neck normally averages height then halves width. Narrow,
    tall images may contain only 12 content steps at width/8. This branch
    retains twice as many without changing the encoder or its weight shapes.
    """
    def __init__(self, architecture, upstream):
        super().__init__(architecture, upstream)
        self.network.decoder.svtr_encoder.use_pool = False
        self.height_score = nn.Conv2d(self.network.encoder.out_channels, 1, 1)
        nn.init.zeros_(self.height_score.weight)
        nn.init.zeros_(self.height_score.bias)

    @classmethod
    def from_base(cls, base, upstream):
        model = cls(base.architecture_config, upstream)
        model.network.load_state_dict(base.network.state_dict(), strict=True)
        return model

    def sequence_features(self, images):
        features = self.network.encoder(images)
        scores = self.height_score(features).float().softmax(dim=2)
        pooled = (features.float() * scores).sum(dim=2, keepdim=True)
        features = self.network.decoder.svtr_encoder(pooled)
        return features.flatten(2).transpose(1, 2)


class SVTRForegroundCaptcha(SVTRCaptcha):
    """Auxiliary renderer-supervised foreground map with a neutral residual gate."""
    def __init__(self, architecture, upstream):
        super().__init__(architecture, upstream)
        self.foreground_head = nn.Sequential(
            nn.Conv2d(self.network.encoder.out_channels, 64, 3, padding=1),
            nn.GELU(), nn.Conv2d(64, 1, 1))
        nn.init.zeros_(self.foreground_head[-1].weight)
        nn.init.zeros_(self.foreground_head[-1].bias)

    @classmethod
    def from_base(cls, base, upstream):
        model = cls(base.architecture_config, upstream)
        model.network.load_state_dict(base.network.state_dict(), strict=True)
        return model

    def forward_with_mask(self, images):
        features = self.network.encoder(images)
        mask_logits = self.foreground_head(features)
        # Starts at exactly one. A wrong foreground estimate cannot zero out
        # recognition features: the multiplier remains in [0.5, 1.5].
        probability = mask_logits.sigmoid()
        centered = probability - probability.mean(dim=(2,3), keepdim=True)
        gated = features * (1 + .5 * centered)
        features = self.network.decoder.svtr_encoder(gated)
        features = features.flatten(2).transpose(1, 2)
        return self.network.decoder.fc(features), mask_logits

    def forward(self, images):
        return self.forward_with_mask(images)[0]


class SVTRForegroundFineCaptcha(SVTRForegroundCaptcha):
    """Fuse early spatial detail with deep context for foreground supervision."""
    def __init__(self, architecture, upstream):
        super().__init__(architecture, upstream)
        channels=self.network.encoder.out_channels+architecture['Encoder']['dims'][0]
        self.foreground_head=nn.Sequential(nn.Conv2d(channels,64,3,padding=1),nn.GELU(),
            nn.Conv2d(64,64,3,padding=1),nn.GELU(),nn.Conv2d(64,1,1))
        nn.init.zeros_(self.foreground_head[-1].weight)
        nn.init.zeros_(self.foreground_head[-1].bias)

    def forward_with_mask(self, images):
        encoder=self.network.encoder
        features,size=encoder.pope(images)
        early=features
        for stage in encoder.stages:
            features,size=stage(features,size)
        context=torch.nn.functional.interpolate(features,size=early.shape[-2:],mode='bilinear',align_corners=False)
        mask_logits=self.foreground_head(torch.cat([early,context],dim=1))
        probability=torch.nn.functional.adaptive_avg_pool2d(mask_logits.sigmoid(),features.shape[-2:])
        centered=probability-probability.mean(dim=(2,3),keepdim=True)
        gated=features*(1+.5*centered)
        features=self.network.decoder.svtr_encoder(gated).flatten(2).transpose(1,2)
        return self.network.decoder.fc(features),mask_logits
