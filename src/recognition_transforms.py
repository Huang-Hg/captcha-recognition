"""Checkpoint-owned inference transforms; legacy checkpoints keep legacy resize."""
from PIL import Image
from torchvision import transforms


class FitCanvas:
    def __init__(self, height=64, width=320):
        if min(height, width) < 1:
            raise ValueError('Canvas dimensions must be positive')
        self.height, self.width = height, width

    def __call__(self, image):
        image = image.convert('RGB')
        scale = min(self.height / image.height, self.width / image.width)
        resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                               Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.width, self.height), 'white')
        canvas.paste(resized, (0, (self.height - resized.height) // 2))
        return canvas


def transform_from_checkpoint(checkpoint):
    mode = checkpoint.get('preprocessing', 'fixed_resize_rgb_normalize_0.5')
    height, width = checkpoint.get('input_size', [64, 192])
    if mode == 'svtr_bgr_aspect_pad':
        from svtr_adapter import SVTRInput
        if min(height, width)<16 or height%16 or width%16:
            raise ValueError('Invalid checkpoint-owned SVTR geometry')
        return SVTRInput(height, width)
    if mode == 'svtr_bgr_48_zero_pad':
        from svtr_adapter import SVTRInput
        if [height, width] != [48, 320]:
            raise ValueError('Unsupported SVTR input geometry')
        return SVTRInput(height, width)
    if mode == 'aspect_fit_white_canvas':
        resize = FitCanvas(height, width)
    elif mode == 'fixed_resize_rgb_normalize_0.5':
        resize = transforms.Resize((height, width))
    else:
        raise ValueError(f'Unsupported preprocessing: {mode}')
    return transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
