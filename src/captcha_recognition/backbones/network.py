"""Build the checkpoint's visual network without an external model registry."""
import copy
from torch import nn
from .svtr_encoder import SVTRv2LNConvTwo33
from .ctc_decoder import CTCDecoder

SOURCE_REVISION = '1ccfc6ee6161f7133b192e16af3a9265273997ba'


class SVTRNetwork(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        config = copy.deepcopy(architecture)
        if config.get('Transform') is not None:
            raise ValueError('The bundled visual network does not support a transform module')
        encoder = config.get('Encoder', {})
        decoder = config.get('Decoder', {})
        if encoder.pop('name', None) != 'SVTRv2LNConvTwo33':
            raise ValueError('Unsupported bundled encoder; expected SVTRv2LNConvTwo33')
        if decoder.pop('name', None) != 'CTCDecoder':
            raise ValueError('Unsupported bundled decoder; expected CTCDecoder')
        if not decoder.get('svtr_encoder') or decoder.get('mid_channels') is not None:
            raise ValueError('Checkpoints require an SVTR neck and a single CTC classifier')
        encoder['in_channels'] = config.get('in_channels', 3)
        self.encoder = SVTRv2LNConvTwo33(**encoder)
        decoder['in_channels'] = self.encoder.out_channels
        self.decoder = CTCDecoder(**decoder)
        # Joint training must allow CTC gradients to reach the visual encoder.
        self.decoder.svtr_encoder.use_guide = False

    def forward(self, images, data=None):
        return self.decoder(self.encoder(images), data=data)
