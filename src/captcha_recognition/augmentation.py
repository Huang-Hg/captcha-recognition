"""Training-only, label-preserving style variation with paired mask geometry."""
import math
import torch
import torch.nn.functional as F


class TrainingAugment:
    def __init__(self, probability=.5):
        if not 0 <= probability <= 1:
            raise ValueError('Augmentation probability must be in [0, 1]')
        self.probability = probability

    def __call__(self, images, masks, widths, generator):
        if not self.probability:
            return images, masks
        count, _, height, width = images.shape
        def rand(*shape):
            return torch.rand(*shape, device=images.device, generator=generator)
        enabled = rand(count) < self.probability
        xs = torch.arange(width, device=images.device).float()[None, None, :]
        ys = torch.linspace(-1, 1, height, device=images.device)[None, :, None]
        valid = xs < widths[:, None, None]
        local_x = xs / (widths[:, None, None]-1).clamp_min(1)*2-1
        scale_x = .92+.04*rand(count, 1, 1)
        scale_y = .92+.04*rand(count, 1, 1)
        shear = .04*(rand(count, 1, 1)-.5)
        amplitude = .02*rand(count, 1, 1)
        phase = 2*math.pi*rand(count, 1, 1)
        source_x = (local_x-shear*ys)/scale_x
        source_y = (ys-amplitude*torch.sin(math.pi*local_x+phase))/scale_y
        grid_x = (source_x+1)*(widths[:, None, None]-1)/(width-1)-1
        grid = torch.stack((grid_x.expand(-1, height, -1), source_y.expand(-1, -1, width)), dim=-1)
        # An explicit support mask keeps padding neutral and avoids sampling beyond
        # each unpadded image's right boundary as if it were real content.
        support = (source_x.abs() <= 1) & (source_y.abs() <= 1) & valid
        warped = F.grid_sample(images, grid, align_corners=True, padding_mode='zeros')
        warped_masks = F.grid_sample(masks.float(), grid, align_corners=True, padding_mode='zeros')
        order = torch.argsort(rand(count, 3), dim=1)
        channels = warped.gather(1, order[:, :, None, None].expand_as(warped))
        shuffled = torch.where((rand(count, 1, 1, 1)<.35), channels, warped)
        gray = channels[:, :1].expand_as(shuffled)
        styled = torch.where(rand(count, 1, 1, 1)<.2, gray, shuffled)
        styled = torch.where(rand(count, 1, 1, 1)<.2, -styled, styled)
        styled = styled*(.8+.4*rand(count, 3, 1, 1)) + .15*(2*rand(count, 3, 1, 1)-1)
        styled = styled.clamp(-1, 1)*support[:, None]
        warped_masks = warped_masks*support[:, None]
        # Preserve untouched examples bit for bit; the dataset cache is never modified.
        return (torch.where(enabled[:, None, None, None], styled, images),
                torch.where(enabled[:, None, None, None], warped_masks, masks.float()))
