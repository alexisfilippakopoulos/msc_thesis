import random
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.distributions as D


class M3(nn.Module):
    def __init__(
        self,
        p_aug: float = 1.0,
        n_modalities: int = 3,
        p_mod: List[float] = [0.25, 0.25, 0.25],
        dim: List[str] = ["time", "time", "time"],
        striped: List[bool] = [False, False, False],
        strategy: str = "sample-wise",
        all_modal: bool = True,
        p_mask: List[float] = [0.2, 0.2, 0.2],
    ):
        """M^3 Front-End layer implementation

        Decides whether to apply or not Masking at a given sample and applies
        Multimodal Augmentations respectively

        Args:
            p_aug (float):
                mask/drop/aug probability, 1-p is the prob to leave the
                sequence unaffected. Tune it by fixing p_mod to 0.33 and p_mask
                at some low value, e.g., 0.1.
            n_modalities (int): number of modalities
            p_mod (List[float]): Per modal augmentation rate
            dim (List[str]): masking type per modality
                - "time": time-masking
                - "feature": feature-masking
            striped (List[bool]): list of boolean attributes
                - True: use striped version
                - False: use random version
            strategy (str):
                - "sample-wise": apply/not aug per sample
                - "batch-wise": apply/not aug per sample
            all_modal (bool):
                - True: apply masks at all modals
                - False: apply masks at only one modal at the time
            p_mask (List[float]): a list of float probabilities which are
                controlling the masking rate in each modality. The suggestion
                in M3 paper is to tie this param across modalities
                and experiment with different p_mod after we have fixed the
                p_aug value.
        """
        super(M3, self).__init__()
        self.p_aug = p_aug
        self.n_modalities = n_modalities
        self.strategy = strategy
        self.all_modal = all_modal
        self.p_mod = p_mod
        self.p_mask, self.dim, self.striped = \
            self.init_pmask(p_mask, dim, striped)

        self.masking_layer = []
        for p_m, dim_m, striped_m in zip(self.p_mask, self.dim, self.striped):
            self.masking_layer.append(
                M2(p_mask=p_m, dim=dim_m, striped=striped_m)
            )

    def init_pmask(self, p_mask, dim, striped):
        init_p_mask = \
            [1.0 / self.n_modalities for _ in range(self.n_modalities)]
        init_dim = ["time" for _ in range(self.n_modalities)]
        init_striped = [False for _ in range(self.n_modalities)]
        if p_mask is not None:
            init_p_mask = p_mask
        if dim is not None:
            init_dim = dim
        if striped is not None:
            init_striped = striped
        return init_p_mask, init_dim, init_striped

    def forward(self, x_text, x_audio=None, x_vision=None, real_len=None):
        mods = [x_text]
        if x_audio is not None:
            mods.append(x_audio)
        if x_vision is not None:
            mods.append(x_vision)
        # List of [B, L, D]

        if self.training:
            if self.strategy == "sample-wise":
                bsz = mods[0].shape[0]
                print(f"bsz is {bsz}")
                if self.all_modal:
                    for m in range(len(mods)):
                        tmp_mods = \
                            self.masking_layer[m](mods[m], real_len=real_len)
                        # how many samples within the batch
                        p_samples=torch.tensor(self.p_mod[m]*self.p_aug).to(mods[m].device)
                        which_sample = D.bernoulli.Bernoulli(probs=p_samples)
                        sample_mask = which_sample.sample((bsz, 1, 1))
                        mods[m] = \
                            tmp_mods * sample_mask + (1-sample_mask) * mods[m]
                else:
                    # (bsz, 3)
                    which_sample = D.bernoulli.Bernoulli(
                        probs = torch.tensor(self.p_aug).to(mods[0].device)
                    )
                    sample_mask = which_sample.sample((bsz, 1, 1))
                    which_modal = \
                        D.one_hot_categorical.OneHotCategorical(
                            torch.tensor(self.p_mod).to(mods[0].device)
                        )
                    modal_mask = which_modal.sample((bsz, ))
                    for m in range(len(mods)):
                        tmp_mods = self.masking_layer[m](
                            mods[m],
                            real_len=real_len,
                        )
                        # import pdb; pdb.set_trace()
                        tmp_modal = modal_mask[:, m].unsqueeze(1).unsqueeze(2)
                        # * modal_mask[:, m] * sample_mask \
                        # + (1 - modal_mask[:, m] * sample_mask) * mods[m]
                        mods[m] = tmp_mods * tmp_modal * sample_mask \
                            + (1 - tmp_modal * sample_mask) * mods[m]
            else:
                #### NOT MEANT TO BE USED
                # batch-wise version
                if random.random() < self.p_aug:
                    bsz, seqlen = mods[0].size(0), mods[0].size(1)
                    # mask different modality for each sample in batch
                    if self.all_modal: # mask all modals
                        for m in range(len(mods)):
                            p_samples = \
                                torch.tensor(self.p_mod[m]).to(mods[m].device())
                            which_sample = D.bernoulli.Bernoulli(probs=1-p_samples)
                            sample_mask = which_sample.sample((bsz, ))
                            tmp_mods = \
                                self.masking_layer[m](mods[m], real_len=real_len)
                            mods[m] = \
                                tmp_mods * sample_mask + (1-sample_mask) * mods[m]
                    else: # mask a single/different modal 
                        # (bsz, 3)
                        for m in range(len(mods)):
                            # FIXME
                            # this is incorrect
                            which_modal = \
                            D.one_hot_categorical.OneHotCategorical(
                                torch.tensor(self.p_mod).to(mods[0].device)
                            )
                            modal_mask = which_modal.sample((bsz, ))
                            mods[m] = self.masking_layer[m](
                                mods[m],
                                real_len=real_len,
                                p_samples=1.0,
                                p_mask=self.p_mask[m]
                            ) * modal_mask[:, m] + \
                            (1 - modal_mask[:, m]) * mods[m]

        return mods

    def __repr__(self):
        shout = (
            self.__class__.__name__
            + "("
            + "strategy="
            + self.strategy
            + ", p_aug"
            + str(self.p_aug)
            + ", p_mask="
            + str(self.p_mask)
            + ", p_mod="
            + str(self.p_mod)
            + ", all_modal="
            + str(self.all_modal)
            + ")"
        )
        return shout


class M2(nn.Module):
    def __init__(self, p_mask=0.1, dim="time", striped=False):
        """
        Args:
            p_mask (float): probability of masking / area of masking
            dim (str): 'time' or 'feature' masking
            striped (bool): when True masks a stripe of len%=p_mask
        """
        super(M2, self).__init__()
        self.p_mask = p_mask
        self.dim = dim
        self.striped = striped
        self.mask_pdf = \
            D.bernoulli.Bernoulli(probs=1 - torch.tensor(p_mask))

    def forward(self, x_m, real_len=None):
        # input tensor has shape: (B, L, D)
        bsz, seqlen, d_m = x_m.size(0), x_m.size(1), x_m.size(2)
        if self.dim == "time":
            if self.striped:
                total_mask = torch.ones((bsz, seqlen), device=x_m.device)
                for i_smp in range(bsz):
                    stripe_len = \
                        int(real_len[i_smp] * self.p_mask)
                    # stripe_start = torch.randint(
                    #     0, real_len[i_smp] - stripe_len
                    # )
                    stripe_start = random.randint(
                        0, real_len[i_smp] - stripe_len
                    )
                    total_mask[i_smp, stripe_start:stripe_start+stripe_len] = \
                        0.0
            else:
                # Be() draws independently regardless the real_len
                # over all padded len.
                total_mask = self.mask_pdf.sample((bsz, seqlen))
            # (B, D) --> (B, D, 1)
            total_mask = total_mask.unsqueeze(2)
        else: # self.dim == "feature"
            if self.striped:
                stripe_len = int(d_m * self.p_mask)
                total_mask = torch.ones((bsz, d_m), device=x_m.device)
                stripe_start = torch.randint(0, d_m - stripe_len, (bsz, ))
                for i_smp in range(bsz):
                    s_start = stripe_start[i_smp].item()
                    total_mask[i_smp, s_start:s_start+stripe_len] = 0.0
            else: # random feature masking
                total_mask = self.mask_pdf.sample((bsz, d_m))
            # (B, D) --> (B, 1, D)
            total_mask = total_mask.unsqueeze(1)

        return x_m*total_mask


if __name__ == "__main__":
    bsz = 6
    seqlen = 8
    # seqlen = 10
    d_m = 10
    # d_m = 8
    # test M2 implementation
    # p_mask = 0.25
    # dim = "feature"
    # striped = True
    # M2_a = M2(p_mask=p_mask, dim=dim, striped=striped)
    # in_tensor = torch.ones((bsz, seqlen, d_m))
    # # out_tensor = M2_a(in_tensor, real_len=real_len)
    # out_tensor = M2_a(in_tensor)
    # # import pdb; pdb.set_trace()


    # check again MIGHT HAVE A BUG
    # COUNT NUMBER
    dim_str = "time"
    strip_flag = True
    masking_layer = M3(
        p_aug=1.0, n_modalities=3, p_mod=[1.0, 1.0, 1.0],
        dim=[dim_str, dim_str, dim_str],
        striped=[strip_flag, strip_flag, strip_flag],
        strategy="sample-wise",
        all_modal=True,
        p_mask=[0.50, 0.50, 0.50]
        )
    t_tensor = torch.ones((bsz, seqlen, 12))
    a_tensor = torch.ones((bsz, seqlen, 8))
    v_tensor = torch.ones((bsz, seqlen, 10))
    real_len = torch.tensor([6, 8, 9, 10, 8, 7])
    out_tensor = masking_layer(t_tensor, a_tensor, v_tensor, real_len)
    import pdb; pdb.set_trace()
