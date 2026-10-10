"""Hash-checked manifests and shared image/label batching for every decoder."""
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from svtr_adapter import SVTRInput


def read_manifest(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['samples']
    if not rows:
        raise ValueError('Empty dataset')
    return rows


def assert_isolated(splits):
    names = list(splits)
    for index, name in enumerate(names):
        for other in names[index+1:]:
            for key in ('label', 'sha256'):
                if {r[key] for r in splits[name]} & {r[key] for r in splits[other]}:
                    raise ValueError(f'Cross-split {key} overlap: {name}/{other}')


class RecognitionData:
    def __init__(self, rows, vocabulary, config, device, transform=None):
        self.rows, self.vocabulary = rows, vocabulary
        self.labels = [row['label'] for row in rows]
        if any(not label or len(label) > config.max_length for label in self.labels):
            raise ValueError('Label is empty or exceeds configured maximum length')
        self.tokens = vocabulary.encode(self.labels, device)
        transform = transform or SVTRInput(config.height, config.width)
        images, widths = [], []
        masks = torch.zeros(len(rows), 1, config.height, config.width, dtype=torch.float16)
        annotated = []
        for index, row in enumerate(rows):
            path = Path(row['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Image hash mismatch: {path}')
            with Image.open(path) as image:
                width = min(config.width, math.ceil(config.height*image.width/image.height))
                images.append(transform(image))
                geometry = image.size
            widths.append(width)
            annotated.append('mask_path' in row)
            if annotated[-1]:
                mask_path = Path(row['mask_path'])
                if hashlib.sha256(mask_path.read_bytes()).hexdigest() != row['mask_sha256']:
                    raise ValueError('Mask hash mismatch')
                with Image.open(mask_path) as mask:
                    if mask.size != geometry:
                        raise ValueError('Mask/image geometry mismatch')
                    target = cv2.resize(np.asarray(mask.convert('L')), (width, config.height)).astype(np.float32)/255.
                masks[index, 0, :, :width] = torch.from_numpy(target)
        self.images = torch.stack(images).to(device)
        self.widths = torch.tensor(widths, device=device)
        self.lengths = torch.tensor([len(text) for text in self.labels], device=device)
        self.masks = masks.to(device)
        self.annotated = torch.tensor(annotated, device=device)
        self.groups = sorted({row['group'] for row in rows})
        ids = {name: index for index, name in enumerate(self.groups)}
        self.group_ids = torch.tensor([ids[row['group']] for row in rows], device=device)
        self.group_rows = [torch.where(self.group_ids == i)[0] for i in range(len(self.groups))]

    def sample(self, count, generator, probabilities=None):
        device = self.images.device
        probabilities = probabilities if probabilities is not None else torch.ones(len(self.groups), device=device)
        groups = torch.multinomial(probabilities, count, replacement=True, generator=generator)
        selected = torch.empty(count, device=device, dtype=torch.long)
        for group in groups.unique().tolist():
            mask = groups.eq(group)
            indices = self.group_rows[group]
            offsets = torch.randint(len(indices), (int(mask.sum()),), device=device, generator=generator)
            selected[mask] = indices[offsets]
        return selected, groups

    def ctc_target(self, selected):
        lengths = self.lengths[selected]
        labels = self.tokens[selected, 1:]
        mask = torch.arange(labels.shape[1], device=labels.device)[None] < lengths[:, None]
        return labels[mask], lengths
