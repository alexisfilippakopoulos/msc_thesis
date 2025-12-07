"""
UHIO-- Unimodal Head in One
"""
import random
import torch.nn as nn

from .singleTask import *
from .subNets.LinearHead import LinearHead

__all__ = ['UHIO']

class UHIO(nn.Module):
    def __init__(self, args, modal='T'):
        super(UHIO, self).__init__()

        # unimodal head
        if args['model_name'] == "mult":
            dst_feature_dims, _ = args.dst_feature_dim_nheads
            output_dim = args.num_classes if args.train_mode == "classification" else 1
            # same dimension for all modalities in MulT implementation
            # import pdb; pdb.set_trace()
            self.head = LinearHead(2*dst_feature_dims, output_dim) 
        elif args['model_name'] == "self_mm":
            if modal == 'T':
                self.head = LinearHead(args.post_text_dim, 1)
            elif modal == 'A':
                self.head = LinearHead(args.post_audio_dim, 1)
            elif modal == 'V':
                self.head = LinearHead(args.post_video_dim, 1)
            else:
                raise KeyError('Not a valid modal given')
        else:
            raise KeyError(f"Not found model with name {args['model_name']}")

    def forward(self, x):
        return self.head(x)