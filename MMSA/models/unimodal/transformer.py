import torch
import torch.nn as nn

from MMSA.models.subNets.transformers_encoder.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer
)

"""need to see where the positional embeddings are going to be added here
"""

class Enc_Transformer(nn.Module):
    def __init__(self, args):
        super(Enc_Transformer, self).__init__()
        output_dim = \
            args.num_classes if args.train_mode == "classification" else 1
        self.project = nn.Conv1d(args.feature_dim, args.embed_dim,
                                 kernel_size=1,
                                 padding=0, bias=False)
        self.enc = TransformerEncoder(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            layers=max(args.layers, 2),
            attn_dropout=args.attn_dropout,
            relu_dropout=args.relu_dropout,
            res_dropout=args.res_dropout,
            embed_dropout=args.embed_dropout,
            attn_mask=args.attn_mask,
            position_embedding=args.positional_embedding
        )
        self.out = nn.Linear(args.embed_dim, output_dim)

    def forward(self, x):
        x = self.enc(self.project(x))
        return self.out(x)

