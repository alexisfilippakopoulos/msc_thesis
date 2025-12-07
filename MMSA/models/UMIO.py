"""
UMIO -- Unimodal Models in One
"""
import torch.nn as nn

from unimodal.transformer import Enc_Transformer
from MMSA.transformations.combine import AugNet


class UMIO(nn.Module):
    def __init__(self, args):
        super(UMIO, self).__init__()
        self.MODEL_MAP = {
            # single-task
            'enc_transformer': Enc_Transformer
        }
        # adding input-feature-augmentation
        self.input_aug = args.get('input_augmentation', False)
        if self.input_aug:
            aug_args = args['augmentation']
            self.augNet = AugNet(aug_args, aug_args['name'])
        # unimodal-model
        lastModel = self.MODEL_MAP[args['model_name']]
        self.Model = lastModel(args)

    def forward(self, x_m, *args, **kwargs):
        if self.input_aug and self.training:
            if 'real_len' in kwargs:
                real_len = kwargs.pop('real_len')
            else:
                raise ValueError(
                    "You should also provide pad_len as keyword argument"
                    )
            x_m = self.augNet(x_m, pad_len=real_len, m_ra=1)
        return self.Model(x_m, *args, **kwargs)
