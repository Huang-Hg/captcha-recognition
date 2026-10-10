"""Shared spatial vision with independent CTC and causal attention decoders."""
import copy
import math
from pathlib import Path
import torch
from torch import nn
from .config import HybridConfig
from .vocabulary import Vocabulary
from svtr_adapter import SVTRForegroundCaptcha, UPSTREAM_COMMIT


def position_1d(length, width, device):
    frequency = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000.) / width))
    angles = torch.arange(length, device=device).float()[:, None] * frequency[None]
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


def position_2d(height, width, channels, device):
    y = position_1d(height, channels//2, device)[:, None].expand(-1, width, -1)
    x = position_1d(width, channels//2, device)[None].expand(height, -1, -1)
    return torch.cat((y, x), dim=-1).reshape(height*width, channels)


class HybridRecognizer(nn.Module):
    architecture = 'dfcr_hybrid_v1'

    def __init__(self, visual_config, config=None, upstream=None):
        super().__init__()
        self.config = config or HybridConfig()
        self.vocabulary = Vocabulary(self.config.alphabet)
        self.visual_config = copy.deepcopy(visual_config)
        self.visual_config['Decoder']['out_channels'] = len(self.config.alphabet) + 1
        upstream = upstream or Path(__file__).resolve().parents[2] / 'assets/upstream/OpenOCR'
        base = SVTRForegroundCaptcha(self.visual_config, upstream)
        self.encoder = base.network.encoder
        self.foreground = base.foreground_head
        self.ctc_neck = base.network.decoder.svtr_encoder
        self.ctc_head = base.network.decoder.fc
        width = self.config.decoder_width
        self.memory_projection = nn.Sequential(nn.Linear(self.encoder.out_channels, width), nn.LayerNorm(width))
        self.token_embedding = nn.Embedding(len(self.config.alphabet)+3, width, padding_idx=self.vocabulary.pad)
        nn.init.normal_(self.token_embedding.weight, std=width**-.5)
        with torch.no_grad():
            self.token_embedding.weight[self.vocabulary.pad].zero_()
        layer = nn.TransformerDecoderLayer(width, self.config.decoder_heads, width*4,
            dropout=self.config.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.attention_decoder = nn.TransformerDecoder(layer, self.config.decoder_layers, norm=nn.LayerNorm(width))
        self.attention_head = nn.Linear(width, len(self.config.alphabet)+1)

    @classmethod
    def from_legacy(cls, checkpoint, config=None, upstream=None):
        from model_checkpoints import model_from_checkpoint
        if checkpoint['architecture'] != 'svtrv2_foreground_captcha':
            raise ValueError('Hybrid initialization requires the audited foreground SVTR candidate')
        config = config or HybridConfig(alphabet=checkpoint['alphabet'])
        if config.alphabet != checkpoint['alphabet']:
            raise ValueError('Vocabulary transfer must be explicit')
        base = model_from_checkpoint(checkpoint, upstream)
        model = cls(checkpoint['model_config'], config, upstream)
        model.encoder.load_state_dict(base.network.encoder.state_dict(), strict=True)
        model.foreground.load_state_dict(base.foreground_head.state_dict(), strict=True)
        model.ctc_neck.load_state_dict(base.network.decoder.svtr_encoder.state_dict(), strict=True)
        model.ctc_head.load_state_dict(base.network.decoder.fc.state_dict(), strict=True)
        return model

    @classmethod
    def from_transfer(cls, checkpoint, config=None, upstream=None):
        if checkpoint.get('architecture') != cls.architecture:
            return cls.from_legacy(checkpoint, config, upstream)
        previous = HybridConfig(**checkpoint['hybrid_config'])
        config = config or previous
        for key in previous.to_dict():
            if key not in ('height', 'width') and getattr(previous, key) != getattr(config, key):
                raise ValueError(f'Unsupported transfer contract change: {key}')
        # Geometry uses dynamic positional encodings; it changes no learned tensor shapes.
        restored = load_hybrid(checkpoint, upstream)
        model = cls(checkpoint['model_config'], config, upstream)
        model.load_state_dict(restored.state_dict(), strict=True)
        return model

    def freeze_statistics(self):
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()

    def visual_parameters(self):
        for module in [self.encoder, self.foreground, self.ctc_neck, self.ctc_head]:
            yield from module.parameters()

    def attention_parameters(self):
        for module in [self.memory_projection, self.token_embedding, self.attention_decoder, self.attention_head]:
            yield from module.parameters()

    def adapt_features(self, features, images, valid_widths):
        return features

    def encode(self, images, valid_widths=None):
        features = self.encoder(images)
        mask_logits = self.foreground(features)
        probability = mask_logits.sigmoid()
        gated = features * (1 + .5 * (probability-probability.mean((2, 3), keepdim=True)))
        gated = self.adapt_features(gated, images, valid_widths)
        ctc = self.ctc_head(self.ctc_neck(gated).flatten(2).transpose(1, 2))
        height, width = gated.shape[-2:]
        memory = self.memory_projection(gated.flatten(2).transpose(1, 2))
        memory = memory + position_2d(height, width, memory.shape[-1], memory.device).to(memory.dtype)
        if valid_widths is None:
            valid_widths = torch.full((len(images),), images.shape[-1], device=images.device)
        columns = torch.ceil(valid_widths.float()*width/images.shape[-1]).long().clamp(1, width)
        padding = (torch.arange(width, device=images.device)[None, None] >= columns[:, None, None])
        padding = padding.expand(-1, height, -1).flatten(1)
        return dict(ctc=ctc, memory=memory, padding=padding, foreground=mask_logits)

    def attention(self, memory, padding, tokens):
        hidden = self.token_embedding(tokens) * math.sqrt(self.config.decoder_width)
        hidden = hidden + position_1d(tokens.shape[1], hidden.shape[-1], hidden.device).to(hidden.dtype)
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool).triu(1)
        hidden = self.attention_decoder(hidden, memory, tgt_mask=causal,
            tgt_key_padding_mask=tokens.eq(self.vocabulary.pad), memory_key_padding_mask=padding)
        return self.attention_head(hidden)

    def forward(self, images):
        """Logit-only CTC compatibility; full training/recognition uses explicit methods."""
        return self.encode(images)['ctc']

    def forward_training(self, images, valid_widths, tokens):
        result = self.encode(images, valid_widths)
        result['attention'] = self.attention(result['memory'], result['padding'], tokens[:, :-1])
        return result

    def autoregressive(self, memory, padding):
        vocabulary = self.vocabulary
        tokens = torch.full((len(memory), 1), vocabulary.bos, device=memory.device, dtype=torch.long)
        finished = torch.zeros(len(memory), dtype=torch.bool, device=memory.device)
        for _ in range(self.config.max_length+1):
            logits = self.attention(memory, padding, tokens)
            next_token = logits[:, -1].argmax(-1)
            next_token = torch.where(finished, vocabulary.eos, next_token)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            finished |= next_token.eq(vocabulary.eos)
            if finished.all():
                break
        # A final non-EOS output is a bounded-length truncation, exposed to callers.
        return vocabulary.ar_decode(tokens[:, 1:self.config.max_length+1]), ~finished

    def recognize(self, images, valid_widths=None, mode='hybrid'):
        if mode in ('ctc_multiview', 'hybrid_multiview'):
            return self.recognize_multiview(images, valid_widths, mode=='hybrid_multiview')
        if mode not in ('ctc', 'attention', 'hybrid', 'all'):
            raise ValueError('Unknown decoder mode')
        result = self.encode(images, valid_widths)
        ctc = self.vocabulary.ctc_decode(result['ctc'])
        if mode == 'ctc':
            return [dict(prediction=text, ctc=text, decoder='ctc') for text in ctc]
        attention, truncated = self.autoregressive(result['memory'], result['padding'])
        if mode == 'attention':
            return [dict(prediction=a, ctc=c, attention=a, decoder='attention', truncated=bool(t))
                    for c, a, t in zip(ctc, attention, truncated)]
        candidates, owners = [], []
        for index, (c, a) in enumerate(zip(ctc, attention)):
            choices = list(dict.fromkeys([c, a]))
            choices = [text for text in choices if text] or ['']
            candidates.extend(choices)
            owners.extend([index]*len(choices))
        owner = torch.tensor(owners, device=images.device, dtype=torch.long)
        tokens = self.vocabulary.encode(candidates, images.device)
        ar_logits = self.attention(result['memory'][owner], result['padding'][owner], tokens[:, :-1]).float()
        target = tokens[:, 1:]
        ar_nll = torch.nn.functional.cross_entropy(ar_logits.transpose(1, 2), target,
            ignore_index=self.vocabulary.pad, reduction='none').sum(1)
        lengths = torch.tensor([len(text) for text in candidates], dtype=torch.long)
        flat = torch.tensor([self.vocabulary.ids[c] for text in candidates for c in text], device=images.device, dtype=torch.long)
        ctc_logits = result['ctc'][owner].float().log_softmax(-1).transpose(0, 1)
        ctc_nll = torch.nn.functional.ctc_loss(ctc_logits, flat, torch.full_like(lengths, ctc_logits.shape[0]),
            lengths, blank=self.vocabulary.blank, reduction='none', zero_infinity=False)
        joint = -.5*(ar_nll+ctc_nll)
        output = []
        for index, (c, a) in enumerate(zip(ctc, attention)):
            choices = [i for i, source in enumerate(owners) if source == index]
            winner = max(choices, key=lambda i: float(joint[i]))
            output.append(dict(prediction=candidates[winner], ctc=c, attention=a, decoder='hybrid',
                truncated=bool(truncated[index]), agreement=c == a,
                candidates=[dict(text=candidates[i], log_score=float(joint[i])) for i in choices],
                score_is_calibrated_confidence=False))
        return output

    def recognize_multiview(self, images, valid_widths=None, use_attention=True):
        from .decoding import ctc_mixture_scores
        original = self.encode(images, valid_widths)
        colorful = ((images[:, 0]-images[:, 1]).abs().flatten(1).amax(1)>1e-6) | ((images[:, 0]-images[:, 2]).abs().flatten(1).amax(1)>1e-6)
        indices = colorful.nonzero(as_tuple=False).flatten()
        views = [row[None] for row in original['ctc']]
        if indices.numel():
            selected = images[indices]
            channels = torch.stack([selected[:, c:c+1].expand(-1, 3, -1, -1) for c in range(3)], dim=1)
            channel_widths = valid_widths[indices].repeat_interleave(3) if valid_widths is not None else None
            extra = self.encode(channels.flatten(0, 1), channel_widths)['ctc']
            extra = extra.reshape(len(indices), 3, *extra.shape[1:])
            for index, other in zip(indices.tolist(), extra):
                views[index] = torch.cat((views[index], other))
        attention, truncated = (self.autoregressive(original['memory'], original['padding']) if use_attention
                                else (['']*len(images), torch.zeros(len(images), device=images.device, dtype=torch.bool)))
        texts, owners, scores = [], [], []
        for index, logits in enumerate(views):
            options = self.vocabulary.ctc_decode(logits)
            if use_attention:
                options.append(attention[index])
            options = list(dict.fromkeys(text for text in options if text)) or ['']
            ids = [tuple(self.vocabulary.ids[c] for c in text) for text in options]
            scores.append(ctc_mixture_scores(logits, ids, self.vocabulary.blank))
            texts.extend(options)
            owners.extend([index]*len(options))
        ctc_scores = torch.cat(scores)
        if use_attention:
            owner = torch.tensor(owners, device=images.device)
            tokens = self.vocabulary.encode(texts, images.device)
            logits = self.attention(original['memory'][owner], original['padding'][owner], tokens[:, :-1]).float()
            ar_scores = -torch.nn.functional.cross_entropy(logits.transpose(1, 2), tokens[:, 1:],
                ignore_index=self.vocabulary.pad, reduction='none').sum(1)
            joint = .5*(ctc_scores+ar_scores)
        else:
            joint = ctc_scores
        results = []
        for index in range(len(images)):
            options = [i for i, source in enumerate(owners) if source == index]
            winner = max(options, key=lambda i: float(joint[i]))
            results.append(dict(prediction=texts[winner], decoder='hybrid_multiview' if use_attention else 'ctc_multiview',
                attention=attention[index] if use_attention else None, views_used=len(views[index]),
                truncated=bool(truncated[index]), score_is_calibrated_confidence=False))
        return results

    def checkpoint(self, **metadata):
        return dict(schema_version=2, architecture=self.architecture, model_config=self.visual_config,
            hybrid_config=self.config.to_dict(), alphabet=self.config.alphabet,
            preprocessing='svtr_bgr_aspect_pad', input_size=[self.config.height, self.config.width],
            upstream_commit=UPSTREAM_COMMIT, model_state_dict=self.state_dict(), **metadata)


def load_hybrid(checkpoint, upstream=None):
    if checkpoint.get('schema_version') != 2 or checkpoint.get('architecture') != HybridRecognizer.architecture:
        raise ValueError('Unsupported model checkpoint schema')
    if checkpoint.get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('Unsupported encoder source revision')
    if checkpoint.get('preprocessing') not in ('svtr_bgr_48_zero_pad', 'svtr_bgr_aspect_pad'):
        raise ValueError('Unsupported checkpoint preprocessing')
    config = HybridConfig(**checkpoint['hybrid_config'])
    if checkpoint['alphabet'] != config.alphabet or checkpoint['input_size'] != [config.height, config.width]:
        raise ValueError('Inconsistent checkpoint metadata')
    if checkpoint['preprocessing']=='svtr_bgr_48_zero_pad' and checkpoint['input_size'] != [48, 320]:
        raise ValueError('Legacy preprocessing requires its original geometry')
    model = HybridRecognizer(checkpoint['model_config'], config, upstream)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model
