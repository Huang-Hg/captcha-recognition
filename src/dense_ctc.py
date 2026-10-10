'Legacy DenseNet, squeeze-and-excitation and bidirectional GRU with CTC.\nUses 64x192 inputs, color and geometric augmentation, AdamW and OneCycleLR.\nReport case-sensitive exact match separately from character accuracy.'

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

CHAR_SET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
BLANK = len(CHAR_SET)          # CTC blank is 62, giving 63 output classes
IMG_H, IMG_W = 64, 192


# ============================================================================
# Model
# ============================================================================

class SEBlock(nn.Module):
    'Squeeze-and-excitation channel attention with a reduced hidden dimension.'
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))          # [B,C]
        return x * w[:, :, None, None]


class DenseLayer(nn.Module):
    'BN -> ReLU -> Conv1x1(4k) -> BN -> ReLU -> Conv3x3(k).'
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList(
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    'Compress channels by half, then pool; pool_w=False preserves sequence width.'
    def __init__(self, in_channels: int, pool_w: bool = True):
        super().__init__()
        out_channels = in_channels // 2
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2) if pool_w else (2, 1),
                                 stride=(2, 2) if pool_w else (2, 1))
        self.out_channels = out_channels

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x), inplace=True)))


class DFCRv2(nn.Module):
    'DenseNet (6/6/12/8, growth 24), SE, bidirectional GRU and CTC.\nFor [B,3,64,192], the spatial sizes are 32x96, 16x48, 8x24, 4x12 and 2x12.\nHeight pooling produces 12 sequence steps. T >= 2L-1 permits worst-case\nrepeated labels of length 4 to 6, but does not guarantee sufficient detail.\nThe wide variant preserves additional horizontal resolution.'
    def __init__(
        self,
        num_classes: int = len(CHAR_SET) + 1,   # Include one CTC blank class
        growth_rate: int = 24,
        block_config: tuple = (6, 6, 12, 8),
        rnn_hidden: int = 128,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        c = 64
        blocks = []
        for i, n in enumerate(block_config):
            blocks.append(DenseBlock(n, c, growth_rate))
            c += n * growth_rate
            if i < len(block_config) - 1:
                blocks.append(SEBlock(c))
                trans = TransitionLayer(c, pool_w=(i < 2))   # The final transition reduces height only
                blocks.append(trans)
                c = trans.out_channels
        self.features = nn.Sequential(*blocks)

        self.bn_final = nn.BatchNorm2d(c)
        self.se_final = SEBlock(c)

        self.rnn = nn.GRU(c, rnn_hidden, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=0.1)
        self.fc = nn.Linear(2 * rnn_hidden, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.features(x)
        x = self.se_final(F.relu(self.bn_final(x), inplace=True))
        x = x.mean(dim=2)                 # Pool height to [B, C, W]
        x = x.permute(0, 2, 1)            # [B, T, C]
        x, _ = self.rnn(x)
        return self.fc(x)                 # [B, T, num_classes] (logits)


# ============================================================================
# Data
# ============================================================================

def build_transforms(train: bool):
    'Apply color, affine and blur augmentation only to training images.'
    base = [
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
    if not train:
        return transforms.Compose(base)
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        transforms.RandomAffine(degrees=8, translate=(0.04, 0.08), scale=(0.9, 1.1),
                                fill=(255, 255, 255)),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


class CaptchaCTCDataset(Dataset):
    'Read labels from filenames; LABEL_index.png avoids case-insensitive collisions.'
    def __init__(self, image_folder: str, transform=None):
        folder = Path(image_folder)
        self.files, self.labels = [], []
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in ('.jpg', '.png', '.jpeg', '.bmp', '.webp'):
                label = p.stem.split('_')[0]
                if all(ch in CHAR_SET for ch in label):
                    self.files.append(p)
                    self.labels.append(label)
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([CHAR_SET.index(c) for c in self.labels[idx]],
                             dtype=torch.long)
        return image, label


class TransformSubset(Dataset):
    'Wrap a random_split subset with a split-specific transform.'
    def __init__(self, subset, transform):
        self.subset, self.transform = subset, transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        base: CaptchaCTCDataset = self.subset.dataset
        real_idx = self.subset.indices[idx]
        image = Image.open(base.files[real_idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor([CHAR_SET.index(c) for c in base.labels[real_idx]],
                             dtype=torch.long)
        return image, label


def collate_ctc(batch):
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    targets = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return images, targets, target_lengths


def ctc_greedy_decode(logits: torch.Tensor) -> list[str]:
    'Greedily decode [B,T,C], collapsing consecutive repeats and removing blanks.'
    pred = logits.argmax(dim=-1)                       # [B,T]
    out = []
    for seq in pred.tolist():
        chars, prev = [], BLANK
        for k in seq:
            if k != prev and k != BLANK:
                chars.append(CHAR_SET[k])
            prev = k
        out.append(''.join(chars))
    return out


# ============================================================================
# Training
# ============================================================================

def evaluate(model, loader, device, autocast_dtype):
    model.eval()
    n_img = n_exact = n_char = n_char_ok = 0
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=autocast_dtype, enabled=device.type == 'cuda'):
                logits = model(images)
            preds = ctc_greedy_decode(logits.float())
            # Reconstruct the label text for each image
            labels, pos = [], 0
            for L in target_lengths.tolist():
                labels.append(''.join(CHAR_SET[i] for i in targets[pos:pos + L].tolist()))
                pos += L
            for p, t in zip(preds, labels):
                n_img += 1
                n_exact += int(p == t)
                n_char += len(t)
                n_char_ok += sum(a == b for a, b in zip(p, t))
    return n_exact / max(n_img, 1), n_char_ok / max(n_char, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data/train')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-lr', type=float, default=3e-3)
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--save-dir', default='./checkpoints_v2')
    ap.add_argument('--seed', type=int, default=3741)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bf16_ok = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f'device={device} amp={autocast_dtype}')

    full = CaptchaCTCDataset(args.data)
    print(f'Total samples: {len(full)}')
    val_size = int(len(full) * args.val_split)
    train_sub, val_sub = random_split(
        full, [len(full) - val_size, val_size],
        generator=torch.Generator().manual_seed(42))
    train_ds = TransformSubset(train_sub, build_transforms(train=True))
    val_ds = TransformSubset(val_sub, build_transforms(train=False))
    print(f'Training: {len(train_ds)}  Validation: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_ctc,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_ctc,
                            pin_memory=True, persistent_workers=args.workers > 0)

    model = DFCRv2().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.15, anneal_strategy='cos')
    ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda', enabled=not bf16_ok and device.type == 'cuda')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for images, targets, target_lengths in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=autocast_dtype, enabled=device.type == 'cuda'):
                logits = model(images)                       # [B,T,C]
            log_probs = logits.float().log_softmax(dim=-1).permute(1, 0, 2)  # [T,B,C]
            input_lengths = torch.full((images.size(0),), logits.size(1),
                                       dtype=torch.long, device=device)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
            running += loss.item() * images.size(0)

        train_loss = running / len(train_ds)
        exact, char_acc = evaluate(model, val_loader, device, autocast_dtype)
        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | '
              f'exact_acc {exact:.4f} | char_acc {char_acc:.4f} | '
              f'lr {lr_now:.2e} | {time.time() - t0:.1f}s', flush=True)

        if exact > best_acc:
            best_acc = exact
            torch.save({'epoch': epoch, 'exact_acc': exact, 'char_acc': char_acc,
                        'model_state_dict': model.state_dict()},
                       save_dir / 'best_model_v2.pth')

    print(f'Training complete, best exact accuracy: {best_acc:.4f}')


# ============================================================================
# Inference interface
# ============================================================================

@torch.no_grad()
def predict_api(model_path: str, image_path: str) -> str:
    # Import lazily: the metadata loader itself imports the model classes.
    from model_checkpoints import model_from_checkpoint
    from recognition_transforms import transform_from_checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = model_from_checkpoint(ckpt).to(device)
    model.eval()
    image = Image.open(image_path).convert('RGB')
    x = transform_from_checkpoint(ckpt)(image).unsqueeze(0).to(device)
    text = ctc_greedy_decode(model(x))[0]
    print(f'Prediction: {text}')
    return text


if __name__ == '__main__':
    main()
