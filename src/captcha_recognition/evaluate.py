"""Evaluate a frozen model and decoder using its checkpoint-owned contracts."""
import argparse
import hashlib
import time
from pathlib import Path
import torch
from .data import RecognitionData, read_manifest
from .evaluation import dump, evaluate
from .inference import Predictor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['ctc', 'ctc_multiview', 'attention', 'hybrid', 'hybrid_multiview', 'all'])
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    if Path(args.output).exists() or args.batch_size < 1:
        raise ValueError('Require a new result file and a positive batch size')
    torch.set_num_threads(4)
    predictor = Predictor(args.checkpoint, mode=None if args.mode=='all' else args.mode)
    if args.mode=='all' and not predictor.checkpoint.get('attention_trained', False):
        raise ValueError('Cannot evaluate untrained attention as a recognition candidate')
    mode = args.mode or predictor.mode
    rows = read_manifest(args.manifest)
    data = RecognitionData(rows, predictor.model.vocabulary, predictor.model.config, predictor.device, predictor.transform)
    start = time.monotonic()
    variants = evaluate(predictor.model, data, args.batch_size, details=True, mode=mode)
    selected = predictor.mode if mode=='all' else mode
    result = dict(**variants[selected], checkpoint=str(predictor.path), checkpoint_sha256=predictor.sha256,
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        architecture=predictor.model.architecture, decoder=selected, precision='fp32', batch_size=args.batch_size,
        seconds=time.monotonic()-start, research_input_size=None)
    if mode=='all':
        result['variants'] = variants
    dump(args.output, result)
    print('EVALUATED', selected, {key: round(value['exact_acc'], 4) for key, value in result['evaluation'].items()}, flush=True)


if __name__ == '__main__':
    main()
