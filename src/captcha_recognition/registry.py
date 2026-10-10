"""Explicit model-family adapters behind one recognition interface."""
from types import SimpleNamespace
import torch
from .model import load_hybrid
from .vocabulary import Vocabulary


class LegacyRecognizer(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from model_checkpoints import model_from_checkpoint
        from recognition_transforms import transform_from_checkpoint
        self.network = model_from_checkpoint(checkpoint)
        self.architecture = checkpoint.get('architecture', 'v2')
        self.vocabulary = Vocabulary(checkpoint['alphabet'])
        height, width = checkpoint.get('input_size', [64, 192])
        self.config = SimpleNamespace(height=height, width=width, alphabet=checkpoint['alphabet'], max_length=40)
        self.transform = transform_from_checkpoint(checkpoint)
        self.available_decoders = ('ctc', 'ctc_multiview')

    def forward(self, images):
        return self.network(images)

    def recognize(self, images, valid_widths=None, mode='ctc'):
        if mode == 'ctc_multiview':
            from multiview_decode import predict_multiview
            return [dict(row, decoder=mode) for row in predict_multiview(self.network, images)]
        if mode != 'ctc':
            raise ValueError('This model does not have an attention decoder')
        return [dict(prediction=text, ctc=text, decoder='ctc') for text in self.vocabulary.ctc_decode(self.network(images))]


def load_recognizer(checkpoint):
    if checkpoint.get('architecture') == 'dfcr_experimental_v1':
        from .experimental import load_experimental
        model = load_experimental(checkpoint)
        model.available_decoders = ('ctc', 'ctc_multiview')
        return model
    if checkpoint.get('architecture') == 'dfcr_hybrid_v1':
        model = load_hybrid(checkpoint)
        model.available_decoders = ('ctc', 'ctc_multiview', 'attention', 'hybrid', 'hybrid_multiview') if checkpoint.get('attention_trained') else ('ctc', 'ctc_multiview')
        return model
    return LegacyRecognizer(checkpoint)
