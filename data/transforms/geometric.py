import random
import torch

from ._base import register_transform

@register_transform('subtract_center_of_mass')
class SubtractCOM(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        pos = data['pos_atoms']
        mask = data['mask_atoms']
        if mask is None or mask.sum() == 0:
            center = torch.zeros(3, dtype=pos.dtype, device=pos.device)
        else:
            center = pos[mask].mean(dim=0)
        data['pos_atoms'] = pos - center.view(1, 1, 3)
        return data
    
