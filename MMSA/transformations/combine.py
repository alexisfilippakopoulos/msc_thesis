import torch
import torch.nn as nn

from MMSA.transformations.pool.mmaug import SoftPerm
# from MMSA.transformations.pool.mmaug_modal import FeRe_modal
# from MMSA.transformations.pool.mmaug_stripe import FeRe_Stripe


class MMAugNet(nn.Module):
    def __init__(self, args, mode):
        super(MMAugNet, self).__init__()
        if mode == 'soft_perm':
            self.augmentation = SoftPerm(
                p_t_mod=args.p_t_mod,
                alpha=[args.alpha_t, args.alpha_a, args.alpha_v],
                maxlen=args.maxlen
            )
        else:
            raise NotImplementedError(
                "Augmentation mode. Pls check your configuration"
                )

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        text_x, audio_x, video_x = \
            self.augmentation([text_x, audio_x, video_x], real_len=kwargs['real_len'])
        return text_x, audio_x, video_x
