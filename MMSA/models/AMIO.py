"""
AMIO -- All Model in One
"""
import random
import torch.nn as nn

from .singleTask import *
from .subNets import AlignSubNet


class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            # unimodal
            'uni': UNI,
            # single-task
            'mms2s': MMSeq2Seq,
            'bienc': BI_ENC,
            'msalm': MSALM,
            'tafc_o1': TAFC_O1,
            'tafc_o4': TAFC_O4,
            'tafc_o5': TAFC_O5,
        }
        self.need_model_aligned = args.get('need_model_aligned', None)
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()

        # multimodal-model
        lastModel = self.MODEL_MAP[args['model_name']]
        self.Model = lastModel(args)
    # def set_mmt(self, verbose=True):
    #     self.Model.set_mmt(verbose)

    # def init_mmt(self, verbose=True):
    #     self.Model.init_mmt(verbose)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)

        return self.Model(text_x, audio_x, video_x, *args, **kwargs)

    def forward_w_mix(self, text_x, audio_x, video_x, *args, **kwargs):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)

        return self.Model.forward_w_mix(text_x, audio_x, video_x, *args, **kwargs)

    def forward_aug(self, text_x, audio_x, video_x, *args, **kwargs):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)

        return self.Model.forward_aug(text_x, audio_x, video_x, *args, **kwargs)

    def vnl_forward(self, text_x, audio_x, video_x, *args, **kwargs):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)

        return self.Model.vnl_forward(text_x, audio_x, video_x, *args, **kwargs)
