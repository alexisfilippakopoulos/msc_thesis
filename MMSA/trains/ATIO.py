"""
ATIO -- All Trains in One
"""
from .singleTask import *
__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # unimodal
            'uni': UNI,
            # single-task
            'mms2s': MMGPT,
            'bienc': BI_ENC,
            'msalm': MSALM,
            'tafc_o1': TAFC_O1,
            'tafc_o4': TAFC_O4,
            'tafc_o5': TAFC_O5,
        }

    def getTrain(self, args):
        print(f"ongoing with {args['model_name']}")
        return self.TRAIN_MAP[args['model_name']](args)
