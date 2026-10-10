"""One inference contract owns vocabulary, preprocessing and selected decoding."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image
import torch
from svtr_adapter import SVTRInput
from .registry import load_recognizer
from .evaluation import dump


class Predictor:
    def __init__(self, checkpoint, device=None, mode=None, expected_sha256=None):
        self.path = Path(checkpoint).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if expected_sha256 is not None and expected_sha256 != self.sha256:
            raise ValueError('Checkpoint hash mismatch')
        self.checkpoint = torch.load(self.path, map_location='cpu', weights_only=True)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = load_recognizer(self.checkpoint).to(self.device).eval()
        self.mode = mode or self.checkpoint.get('selected_decoder', 'ctc')
        self.mode = {'greedy': 'ctc', 'rgb_channel_ctc_mixture': 'ctc_multiview'}.get(self.mode, self.mode)
        if self.mode not in self.model.available_decoders:
            raise ValueError('Unsupported inference decoder')
        self.transform = getattr(self.model, 'transform', None) or SVTRInput(self.model.config.height, self.model.config.width)

    @torch.inference_mode()
    def predict(self, images):
        tensors = torch.stack([self.transform(image) for image in images]).to(self.device)
        widths = torch.tensor([math.ceil(self.model.config.height*image.width/image.height) for image in images], device=self.device)
        return self.model.recognize(tensors, widths, self.mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sources = ap.add_mutually_exclusive_group(required=True)
    sources.add_argument('--checkpoint')
    sources.add_argument('--candidate')
    ap.add_argument('--image', required=True, nargs='+')
    ap.add_argument('--mode', choices=['ctc', 'ctc_multiview', 'attention', 'hybrid', 'hybrid_multiview'])
    ap.add_argument('--device', choices=['cpu', 'cuda'])
    ap.add_argument('--output')
    args = ap.parse_args()
    torch.set_num_threads(4)
    expected = None
    config = None
    if args.candidate:
        file = Path(args.candidate).resolve()
        config = json.loads(file.read_text(encoding='utf-8'))
        path = Path(config['checkpoint'])
        args.checkpoint = path if path.is_absolute() else file.parent.parent/path
        expected = config['checkpoint_sha256']
        args.mode = args.mode or config['decoder']
    predictor = Predictor(args.checkpoint, args.device, args.mode, expected)
    if config:
        for key in ('architecture', 'alphabet', 'input_size', 'preprocessing', 'experimental_variant'):
            if key in config and config[key] != predictor.checkpoint.get(key):
                raise ValueError(f'Candidate metadata mismatch: {key}')
    images = []
    for filename in args.image:
        with Image.open(filename) as image:
            images.append(image.convert('RGB'))
    rows = predictor.predict(images)
    for filename, row in zip(args.image, rows):
        row['image'] = str(Path(filename).resolve())
    result = dict(architecture=predictor.model.architecture, checkpoint_sha256=predictor.sha256,
                  decoder=predictor.mode, precision='fp32', predictions=rows)
    if args.output:
        dump(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
