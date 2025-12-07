'''
This is the implementation used in the ICASSP submission of MMAug.
Some hyperparameters which are redundant are set to their default value
in this implementation. Moreover, the padding is at the end (suffix) so the
implementation takes into account this fomat. The overall pipeline is
identical to not affect any results.
'''
import random
from typing import List, Optional, Dict

import torch
import numpy as np
import torch.nn as nn
import torch.distributions.beta as beta
import torch.distributions as D


class FeRe_modal(nn.Module):
    def __init__(
        self,
        p_modal: (List[float]) = [0.33, 0.33, 0.33],
        p_t_mod: (List[float]) = [1.0, 1.0, 1.0],
        mask_dim: bool = True,
        alpha: (List[float]) = [0.0],
        permute: str = "time",
        replacement: bool = False,
        time_window: Optional[str] = "uniform",
        window_len: Optional[int] = -1,
        maxlen: Optional[int] = 50,
        mixup: Optional[bool] = False,
        discard_zero_pad: Optional[bool] = True,
        constant_val: Optional[float] = -1.0,
        use_beta: Optional[bool] = True,
        single_pick: Optional[bool] = False,
    ):
        """
        Modified implementation of FeRe in which only one modality at a time is being picked

        Args:
            p_modal (List[float]): Probability of augmenting each modality
            p_t_mod (List[float]): Resample probabilities for each modality across timesteps.
                default=1's which means that we resample at every timestep
            mask_dim (bool): copy the same "dimension" across all the timesteps of a given 
                modality. Induces time-invariance. Default: True
            alpha (List[float]): used in sampling uniform distribution for each modality.
                Encodes the amount of features to be resampled. In order to resample the same
                distribution one should give a list of multiple variables as input in [0,1).
            permute (str): which tensor "dimension" is going to be permuted to produce the tensor
                that is going to be pasted in the original one. Choices are "time" and "batch".
                - "time": induces the same label and simply shuffles the time-order of the tensor 
                iteself
                - "batch": respects the time order but mixes samples from different labels
            replacement (bool): determines whether to use replacement when shuffling 
                the sequence itself. True means that some features may be copied more than once
                whereas False denotes that once copied you cannot be copied again.
            time_window (str): "uniform" and "hann" available choices for neighborhood sampling
            window_len (int): number of total neighbors, assumed even, i.e. symmetric
            maxlen (int): must be given as input and refers to the maximum sequence length
                In this implementation we also assume that the sequences are already zero-padded
            mixup (bool): handles whether to use cutmix-like (FeRe) or mixup-like, default:False,
                Mixup might work better at intermed represesntations.
            discard_zero_pad (bool): when true resamples only from the real length sequence
                and ignores the zero padded part at the beginning
            constant_val (float): A constant value, i.e. distribution which is used to sample from
                Default: -1.0, when negative it skips this constant value. Suggested values are
                0.0 and 1.0. When 0.5, denotes random noise in with zero mean and 0.5 std
            use_beta (bool): when true samples from a Betta distribution for each one
                of the involved modalities, otherwise from a uniform
            single_pick (bool): when true samples from a Betta distribution for each one
                of the involved modalities, otherwise from a uniform
        """
        super(FeRe_modal, self).__init__()
        self.p_t_mod = p_t_mod
        self.n_modalities = len(self.p_t_mod)
        self.mask_dim = mask_dim
        self.alpha = alpha
        self.permute = permute
        self.replacement = replacement
        self.time_window = time_window
        self.window_len = window_len
        self.maxlen = maxlen
        self.mixup = mixup
        self.discard_zero_pad = discard_zero_pad
        self.constant_val = constant_val
        self.use_beta = use_beta
        # beta version feature
        # under testing
        self.single_pick = single_pick
        # import pdb; pdb.set_trace()
        # neighbourhood sampling
        if self.window_len > 0:
            self.step_weights = \
                self.get_timestep_weights()
            self.relative_idx = \
                torch.arange(-self.window_len//2, self.window_len//2 + 1, step=1)
            self.absolute_idx = \
                torch.arange(0, self.maxlen)

        # per modality uniform distribution
        # self.beta_mod = []
        # for feat_alpha in self.alpha:
        #     self.beta_mod.append(D.uniform.Uniform(feat_alpha, 1.0))
        self.beta_mod = self.set_beta_mod()

        self.which_modal = D.one_hot_categorical.OneHotCategorical(
            torch.tensor(p_modal)
        )

        # OPTIONAL for the time
        # distribution which defines which timesteps 
        # are going to be blended (time-oriented mask)
        self.time_distribution = []
        for k in range(self.n_modalities):
            self.time_distribution.append(
                self.get_bernoulli_distribution(float(self.p_t_mod[k]))
            )

    def set_beta_mod(self):
        beta_mod = []
        for feat_alpha in self.alpha:
            if self.use_beta:
                print("------------using BEta ------------")
                beta_mod.append(D.beta.Beta(feat_alpha, feat_alpha))
            else:

                beta_mod.append(D.uniform.Uniform(feat_alpha, 1.0))
        print(f"Beta mod is {beta_mod}")
        return beta_mod

    def reset_fere_alpha(self, alpha, verbose=True):
        "alpha here is a list of three values"
        if verbose:
            print(f"Changing fere-alpha from {self.alpha} to {alpha}")
        self.alpha = alpha
        self.beta_mod = self.set_beta_mod()

    def get_timestep_weights(self):
        weights = torch.ones((self.maxlen, self.window_len))
        for i in range(self.maxlen):
            if i - self.window_len//2 < 0:
                weights[i, :(self.window_len//2 - i)] = 0
            if (i + self.window_len // 2) > self.maxlen - 1:
                weights[i, (self.maxlen-1-i-self.window_len//2):] = 0
        if self.time_window == "uniform":
            # import pdb; pdb.set_trace()
            return weights / torch.sum(weights, dim=1).unsqueeze(1)
        else:
            # hann case
            # import pdb; pdb.set_trace()
            hann_weights = \
                    torch.hann_window(window_length=self.window_len + 2,
                                      periodic=False)[1:-1]
            weights = weights * hann_weights
            return weights / torch.sum(weights, dim=1).unsqueeze(1)

    @staticmethod
    def get_bernoulli_distribution(zero_out_prob):
        '''defines a distribution from which we sample which feature-dimensions are going 
        to be blended for every feature tensor (area-oriented masking)
        Tip: probability of drawing 1 is probs
        '''
        # import pdb; pdb.set_trace()
        return D.bernoulli.Bernoulli(probs = 1 - torch.tensor(zero_out_prob))

    def set_permute(self, p_batch=0.5):
        # import pdb; pdb.set_trace()
        if p_batch >= random.random():
            self.permute = "batch"
        else:
            self.permute = "time"

    def forward(self, mods, m_ra=1, pad_len=None):
        """FeRe forward implementation
        Sample from $n_modalities$ independent Uniform distributions to mask "mask_area"
        for each one of the modalities.
        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations
            m_ra (int): repeated augmentation index, default = 1
            pad_len (torch.Tensor): [B] tensor of ints denoting the preffix padded length of the
                sequence
        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        #  already in a list form as of
        # text, audio, video
        # import pdb; pdb.set_trace()
        # mods = list(mods)

        # List of [B, L, D]
        if self.training:
            # if self.use_beta:
            #     print("pasok")
            bsz, seqlen = mods[0].size(0), mods[0].size(1)
            bsz = int(m_ra * bsz)
            if m_ra > 1 and self.discard_zero_pad:
                pad_len = pad_len.repeat(m_ra)
            copy_area = []
            if len(self.beta_mod) == 1 or self.single_pick:
                # print(f"Ongoing with single pick per modality")
                # get $bsz$ initial values and use them for all modalities
                copy_val = self.beta_mod[0].sample((bsz, ))
                if len(set(self.alpha)) == 1:
                    # there exists only one value
                    copy_area = [copy_val for _ in range(self.n_modalities)]
                else:
                    copy_area.append(copy_val)
                    # This branch auto9matically handles the case of unbalanced rates and 
                    # forces the text rate to be $ge$ than av rate
                    # import pdb; pdb.set_trace()
                    av_rate = D.uniform.Uniform(torch.zeros((bsz, )), copy_val).sample()
                    # av_rate = self.beta_mod[1].sample((bsz, ))
                    # print(f"Text Rate is {copy_val} while AV Rate is {av_rate}")
                    for _ in range(self.n_modalities - 1):
                        copy_area.append(av_rate)
            else:
                aug_modal_mask = self.which_modal.sample((bsz, )) # mask
                for i, _ in enumerate(self.beta_mod):
                    copy_area.append(
                        self.beta_mod[i].sample((bsz, )) * aug_modal_mask[:, i]
                    )

            for i_modal, p_modal in enumerate(self.p_t_mod):
                d_modal = mods[i_modal].size(2)
                # STEP:1 - sample Beta when
                # if len(self.beta_mod) > 1:
                #     # this branch handles the resmapling case 
                #     # with the same or different $alpha$ values
                #     # copy_area : lambda
                #     copy_area.append(self.beta_mod[i_modal].sample((bsz,)))

                # defines a distribution from which we sample 
                # which feature-dimensions are going 
                # to be blended for every feature tensor (area-oriented masking)
                self.area_distribution = \
                    self.get_bernoulli_distribution(copy_area[i_modal].clone().detach().float())

                # zeros
                # TODO: define a zero tensor rather than sampling
                modal_timestep_mask = \
                    self.time_distribution[i_modal].sample((bsz, seqlen)).to(mods[0].device)
                # import pdb; pdb.set_trace()
                modal_timestep_mask = modal_timestep_mask.unsqueeze(2)
                if self.mask_dim:
                    # print("--------------I mask my dim---------------")
                    # if self.patched:
                    #     # print("-------------I go in patches-----------")
                    #     # initialize the mask
                    #     area_mask = torch.ones((bsz, d_modal)).to(mods[0].device)
                    #     # sample from a Bernoulli to count how many features are going to be masked
                    #     # masked_feats = \
                    #     #     self.area_distribution.sample((1, d_modal)).int()
                    #     # import pdb; pdb.set_trace()
                    #     masked_feats = \
                    #         torch.sum(self.area_distribution.sample((1, d_modal)), dim=1).int()
                    #     # count zeros
                    #     masked_feats = d_modal - masked_feats.detach().cpu().numpy() - 1
                    #     masked_feats = \
                    #         np.clip(masked_feats,
                    #                 a_min=0,
                    #                 a_max=d_modal-2)
                    #     random_end = \
                    #         np.random.randint(low=masked_feats, high=d_modal-1).astype(int).reshape(-1)
                    #     random_start = \
                    #         (random_end - masked_feats).astype(int).reshape(-1)
                    #     for i, (rs, re) in enumerate(zip(random_start, random_end)):
                    #         area_mask[i, rs:re] = 0.0
                    #     area_mask = area_mask.unsqueeze(1)
                    #     # import pdb; pdb.set_trace()
                    # else:
                    # keep constant masking across feature dimensions through timesteps
                    area_mask = \
                        self.area_distribution.sample((1, d_modal)).to(mods[0].device)
                    area_mask = area_mask.permute(2, 0, 1)
                else:
                    # general case
                    area_mask = \
                        self.area_distribution.sample((seqlen, d_modal)).to(mods[0].device)
                    area_mask = area_mask.permute(2, 0, 1)
                # Reverse logic because we want to sample masks, i.e. zeros
                # when modal mask is 0 then copy the area mask part
                # when modal mask is 1 then do not mask-> 1's everywhere
                # import pdb; pdb.set_trace()

                if m_ra > 1:
                    mods[i_modal] = mods[i_modal].repeat(m_ra, 1, 1)

                tmp_tensor = mods[i_modal].clone().detach().to(mods[0].device)
                if self.permute == "time":
                    # print("-------------Time Blending-----------")
                    if self.time_window == "uniform":
                        # print("-------------I go in patches-----------")
                        if self.window_len == -1:
                            if self.discard_zero_pad:
                                # print("---------Sample from the real length only-------")
                                multinomial_probs = torch.ones(bsz, seqlen).to(mods[0].device)
                                # normalization
                                # import pdb; pdb.set_trace()
                                # print(pad_len)
                                for k in range(bsz):
                                    # pad len here takes into account front padding
                                    # multinomial_probs[k, :pad_len[k]] = 0
                                    # fo pad len here takes into account back padding
                                    ### THIS IS ACTUALLY THE REAL LENGTH
                                    multinomial_probs[k, pad_len[k]:] = 0
                                # for i, j in enumerate(pad_len):
                                #     if j == 50:
                                #         print(f"found zero len seq at position {i}")
                                #         print(mods[0][i])
                                #         import pdb ; pdb.set_trace()
                                # print(seqlen)
                                ### this is the old implementation for the
                                ### front padded sequence
                                # multinomial_probs = \
                                #     multinomial_probs / (seqlen - pad_len.view(-1, 1))
                                ### this is the new implementation for the
                                ### front padded sequence
                                ### probability normalization
                                multinomial_probs = \
                                    multinomial_probs / pad_len.view(-1, 1)
                                # import pdb; pdb.set_trace()
                                # print(torch.sum(multinomial_probs, dim=1))
                                # print(multinomial_probs)
                                randperm_mask = \
                                    torch.multinomial(
                                        multinomial_probs,
                                        seqlen,
                                        replacement=self.replacement
                                    ).to(mods[0].device)
                                # randperm outputs a "back" padded mask -> need to flip
                                # randperm_mask = torch.flip(randperm_mask, [1])
                                randperm_mask = \
                                    randperm_mask \
                                    + torch.arange(start=0, end=bsz).to(mods[0].device).reshape(bsz, 1) * seqlen
                                tmp_tensor = \
                                    tmp_tensor.reshape(-1, d_modal)
                                tmp_tensor = \
                                    tmp_tensor[randperm_mask, :].reshape(bsz, seqlen, d_modal)
                            else:
                                # print("------Sample from the whole sequence-----")
                                randperm_mask = \
                                    torch.multinomial(torch.ones(bsz, seqlen) / seqlen,
                                                    seqlen,
                                                    replacement=self.replacement) \
                                    + torch.arange(start=0, end=bsz).reshape(bsz, 1) * seqlen
                                randperm_mask = \
                                    randperm_mask.to(mods[0].device).reshape(-1)
                                tmp_tensor = \
                                    tmp_tensor.reshape(-1, d_modal)
                                tmp_tensor = \
                                    tmp_tensor[randperm_mask, :].reshape(bsz, seqlen, d_modal)
                        elif self.window_len > 0:
                            # print("-----------sample from a uniform neighborhood-----------")
                            # import pdb; pdb.set_trace()
                            randpick_mask = \
                                D.Categorical(self.step_weights.to(mods[0].device)).sample((bsz, ))
                            tmp_tensor_idx = \
                                self.relative_idx.to(mods[0].device)[randpick_mask] \
                                + self.absolute_idx.to(mods[0].device)
                            # construct a permutation matrix along the batch
                            perm_matrix = torch.eye(self.maxlen).to(mods[0].device)
                            perm_matrix = perm_matrix[tmp_tensor_idx]
                            tmp_tensor = torch.bmm(perm_matrix, tmp_tensor)
                else:
                    randperm_mask = torch.randperm(n=bsz).view(-1)
                    tmp_tensor = tmp_tensor[randperm_mask, :, :]

                if self.constant_val >= 0.0:
                    print("Needs to be fixed for the back paddding scenario")
                    if self.discard_zero_pad:
                        # only mask the real len
                        tmp_tensor = \
                            torch.zeros(mods[i_modal].shape).to(mods[0].device)
                        bsz = mods[0].size(0)
                        for k in range(bsz):
                            if self.constant_val == 0.5:
                                # print("i am here")
                                # FIXME
                                tmp_tensor[k, pad_len[k]:] = \
                                    tmp_tensor[k, pad_len[k]:].data.new(
                                        tmp_tensor[k, pad_len[k]:].size()
                                    ).normal_(0.0, self.constant_val)
                            else:
                                tmp_tensor[k, pad_len[k]:] = self.constant_val
                    else:
                        # this branch masks the whole (padded) sequence
                        if self.constant_val == 0.5:
                            tmp_tensor = \
                                torch.ones(mods[i_modal].shape).to(mods[0].device).normal_(0.0, self.constant_val)
                        else:
                            tmp_tensor = \
                                torch.ones(mods[i_modal].shape).to(mods[0].device) * self.constant_val


                ### cut-paste case
                # when area_mask is 1-> keep that feature else replace it
                # when modal_timestep_mask 1-> use existing feature else replace with blended feat
                if self.mixup:
                    # FIXME: here we need to refine the lamda
                    # a potential solution is to do:
                    # mods[i] = (1 - copy_area)*mods[i] +  copy_area*mods[i]
                    # print("------ intermed layer MixUp -------")
                    # mean_lambda = \
                    #     torch.mean(torch.stack(copy_area), dim=0).unsqueeze(1)
                    copy_area_tensor = copy_area[0].reshape(-1, 1, 1)
                    mods[i_modal] = (1 - copy_area_tensor) * mods[i_modal] \
                                    + copy_area_tensor * tmp_tensor 
                else:
                    # print("------ input  FeRe -----")
                    # cutmix -like approach
                    mods[i_modal] = \
                        (mods[i_modal] * area_mask 
                        + (1 - area_mask) * tmp_tensor) * (1 - modal_timestep_mask) \
                        + mods[i_modal] * modal_timestep_mask
                    # import pdb; pdb.set_trace()
                    # total_modal_mask = torch.logical_or(modal_mask, area_mask).float()
                    # mods[i_modal] = mods[i_modal] * total_modal_mask

            return mods

    def __repr__(self):
        shout = (
            self.__class__.__name__
            + "("
            + "p_mod="
            + str(self.p_t_mod)
            + f", mask_dim={self.mask_dim}"
            + f", alpha={self.alpha}"
            + f", permute={self.permute}"
            + f", reaplacement={self.replacement}"
            + ")"
        )
        return shout
