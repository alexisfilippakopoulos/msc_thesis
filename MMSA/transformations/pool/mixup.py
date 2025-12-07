'''
This script contains the implementations of feature-based mixup and proposed
variants. For each implementation we will refer to the corresponding paper.
'''
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributions.beta as beta
import torch.distributions.normal as normal
import torch.distributions.uniform as uniform
import torch.distributions.dirichlet as dirichlet
import torch.distributions.bernoulli as bernoulli

from typing import List, Optional, Dict, Union


def randperm(x1, x2):
    '''Permutes two tensors according to a common permutation across their
    first (zero) dimension. Assumes batch_first implementations
    Args:
        x1: (B, *, *)
        x2: (B, *)
    '''
    indices = torch.randperm(x1.size(0))
    x1_perm = x1[indices]
    x2_perm = x2[indices]
    return x1_perm, x2_perm


class MixUp(nn.Module):
    def __init__(
            self,
            alpha_1: float = 1.0,
            alpha_2: Optional[Union[float, None]] = None,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01
    ):
        """Replica of the official mixup paper between a pair of embeddings
        and their labels
        https://arxiv.org/abs/1710.09412
        Also optionally uses the random noise trick from
        https://arxiv.org/abs/1805.11272 which interpolates each component with
        a different factor lambda. Similar may be Noisy Feature MixUp paper.

        Args:
            alpha_1 (float): the first alpha (l2r) in the beta distribution
            alpha_2 (float): the second alpha (l2r) in the beta distribution
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
        """
        super(MixUp, self).__init__()
        self.permute = False # when False splits the batch
        self.alpha_1 = alpha_1
        if alpha_2 is None:  # if none overwrite with first alpha
            self.alpha_2 = alpha_1
        else:
            self.alpha_2 = alpha_2  # non-symmetric beta case
        self.add_noise = add_noise
        self.std = std

        # define the beta distribution from which we will sample
        self.beta = beta.Beta(
            torch.tensor(self.alpha_1),
            torch.tensor(self.alpha_2)
            )

        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def forward(self, x, target):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x (torch.tensor): [B, D]
            target (torch.tensor): [B, ]
        """
        if self.training:
            bsz, d = x.size(0), x.size(1)
            if self.permute:
                x_perm, target_perm = self.perm(x, target)  # permute
                # one lambda for every sample
                mixing_factor = self.beta.sample((bsz, )).to(x.device)
                # perform target mixing before optional noise addition
                target = mixing_factor * target + (1 - mixing_factor) * target_perm
                mixing_factor.unsqueeze_(1)
                if self.add_noise:
                    mixing_factor = mixing_factor \
                        + self.additive_noise.sample((bsz, d)).to(x.device)
                    mixing_factor.clamp_(min=0, max=1)
                # mix features
                x = mixing_factor * x + (1 - mixing_factor) * x_perm
            else:
                # bsz should be even for this to work
                x0 = x[:bsz//2]
                x1 = x[bsz//2:]
                y0 = target[:bsz//2]
                y1 = target[bsz//2:]
                # one lambda for half samples
                mixing_factor = self.beta.sample((bsz//2, )).to(x.device)
                # perform target mixing before optional noise addition
                y_mix = mixing_factor * y0 + (1 - mixing_factor) * y1
                mixing_factor.unsqueeze_(1)
                if self.add_noise:
                    mixing_factor = mixing_factor \
                        + self.additive_noise.sample((bsz, d)).to(x.device)
                    mixing_factor.clamp_(min=0, max=1)
                # mix features
                x_mix = mixing_factor * x0 + (1 - mixing_factor) * x1
                x = x_mix
                target = y_mix
        return x, target


class SeqMixUp(nn.Module):
    def __init__(
            self,
            alpha_1: float = 1.0,
            alpha_2: Optional[Union[float, None]] = None,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01
    ):
        """This is an implementation for sequential MixUp which does not take
        into account zero-padding.
        https://arxiv.org/pdf/1905.08941.pdf
        Also optionally uses the random noise trick from
        https://arxiv.org/abs/1805.11272 which interpolates each component with
        a different factor lambda.

        Three choices:
            a) l + l[t,d], every timestep and dimension has different l
            b) l + l[t], every timestep has different lambda
            c) l + l[d], every dimension has different lambda
        We implement a) here.

        Args:
            alpha_1 (float): the first alpha (l2r) in the beta distribution
            alpha_2 (float): the second alpha (l2r) in the beta distribution
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
        """
        super(SeqMixUp, self).__init__()
        self.alpha_1 = alpha_1
        if alpha_2 is None:  # if none overwrite with first alpha
            self.alpha_2 = alpha_1
        else:
            self.alpha_2 = alpha_2  # non-symmetric beta case
        self.add_noise = add_noise
        self.std = std

        # define the beta distribution from which we will sample
        self.beta = beta.Beta(
            torch.tensor(self.alpha_1),
            torch.tensor(self.alpha_2)
            )

        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def forward(self, x, target):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x (torch.tensor): [B, L, D]
            target (torch.tensor): [B, ]
        """
        if self.training:
            bsz, maxlen, d = x.size(0), x.size(1), x.size(2)
            x_perm, target_perm = self.perm(x, target)  # permute
            # one lambda for every sample
            mixing_factor = self.beta.sample((bsz, )).to(x.device)
            # perform target mixinf before optional noise addition
            target = mixing_factor * target + (1 - mixing_factor) * target_perm
            mixing_factor.unsqueeze_(1).unsqueeze_(2)
            if self.add_noise:
                mixing_factor = mixing_factor \
                    + self.additive_noise.sample((bsz, maxlen, d)).to(x.device)
                mixing_factor.clamp_(min=0, max=1)
            # mix features
            print(f"mixing factor {mixing_factor}")
            print(f"x is {x}")
            print(f"x_perm is {x_perm}")
            x = mixing_factor * x + (1 - mixing_factor) * x_perm

        return x, target


class MultiMix(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_low: float = 0.5,
            uni_high: Optional[float] = 2.0,
    ):
        """Replica of the official convex mixup paper. This implementation
        takes into account all the available vectors and does not reweight
        them with attention weights as in the original paper.
        https://arxiv.org/abs/2206.14868

        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
        """
        super(MultiMix, self).__init__()
        self.n_samples = n_samples
        self.uni_low = uni_low
        self.uni_high = uni_high

        self.alpha_pdf = uniform.Uniform(
            low=torch.tensor(self.uni_low),
            high=torch.tensor(self.uni_high),
        )

    def forward(self, x, target):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x (torch.tensor): [B, D]
            target (torch.tensor): [B, ]
        """
        if self.training:
            bsz, d = x.size(0), x.size(1)
            new_bsz = self.n_samples
            # sample alpha's for Dirichlet
            alphas = self.alpha_pdf.sample((bsz, )).to(x.device)
            # (new_bsz, bsz)
            mixing_factors = \
                dirichlet.Dirichlet(alphas).sample((new_bsz,))
            x_new = torch.matmul(mixing_factors, x)
            target_new = torch.matmul(mixing_factors, target)
            x = x_new
            target = target_new
        return x, target


class ConvexMixUp(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_low: float = 0.5,
            uni_high: Optional[float] = 2.0,
            targeted: Optional[bool] = False,
            a_target: Optional[float] = 3.0,
            a_other: Optional[float] = 0.1,
    ):
        """Replica of the official convex mixup paper. This implementation
        takes into account all the available vectors and does not reweight
        them as in the original paper.
        https://arxiv.org/abs/2206.14868

        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
            targeted (bool): by default does not use the targeted approach
                where a target vector is held and the others are interpolated
                via small factors
            a_target (float): the concentration of the target sample, 3,4,5
            a_other (float): the concentration of the other sample, 0.1, 0.3
        """
        super(ConvexMixUp, self).__init__()
        self.n_samples = n_samples
        self.uni_low = uni_low
        self.uni_high = uni_high
        self.a_target = a_target
        self.a_other = a_other
        self.targeted = targeted

        if self.targeted:
            self.a_target = a_target
            self.a_other = a_other
        else:
            self.alpha_pdf = uniform.Uniform(
                low=torch.tensor(self.uni_low),
                high=torch.tensor(self.uni_high),
            )

    def forward(self, x, target):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x (torch.tensor): [B, D]
            target (torch.tensor): [B, ]
        """
        if self.training:
            bsz, d = x.size(0), x.size(1)
            new_bsz = self.n_samples
            if self.targeted:
                # create diagonal
                alphas = (torch.eye(bsz).to(x.device)) \
                    * (self.a_target - self.a_other)
                # expand alphas
                alphas = alphas.repeat(new_bsz//bsz + 1, 1)[:new_bsz]
                # add self.a_other everywhere
                alphas += self.a_other
                # permute to get random places for a_target in the matrix
                perm = torch.randperm(new_bsz).to(x.device)
                alphas = alphas[perm]
                # sample dirichlet, already in (new_bsz, bsz) format
                mixing_factors = dirichlet.Dirichlet(alphas).sample()
                x_new = torch.matmul(mixing_factors, x)
                target_new = torch.matmul(mixing_factors, target)
            else:
                # sample alpha's for Dirichlet
                alphas = self.alpha_pdf.sample((bsz, )).to(x.device)
                # print(alphas)
                # print(f"the shape of x is {x.shape}")
                # (new_bsz, bsz)
                mixing_factors = \
                    dirichlet.Dirichlet(alphas).sample((new_bsz,))
                # print(mixing_factors.shape)
                # print(mixing_factors)
                x_new = torch.matmul(mixing_factors, x)
                target_new = torch.matmul(mixing_factors, target)
            x = x_new
            target = target_new
        return x, target


class SeqConvexMixUp(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_low: float = 0.5,
            uni_high: Optional[float] = 2.0,
            reweight: Optional[str] = "uniform",
            targeted: Optional[bool] = False,
            a_target: Optional[float] = 3.0,
            a_other: Optional[float] = 0.1,
            clf_all: Optional[bool] = False,
    ):
        """Sequential version of the official convex mixup paper.
        https://arxiv.org/abs/2206.14868

        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
            reweight Optional(str): "uniform", "gap", "attn"
            targeted (bool): by default does not use the targeted approach
                where a target vector is held and the others are interpolated
                via small factors
            a_target (float): the concentration of the target sample, 3,4,5
            a_other (float): the concentration of the other sample, 0.1, 0.3
            clf_all (bool): when True return targets for all timesteps/tokens
                in the sequence. When False returns only one (MSA task).
        """
        super(SeqConvexMixUp, self).__init__()
        self.n_samples = n_samples
        self.uni_low = uni_low
        self.uni_high = uni_high
        self.a_target = a_target
        self.a_other = a_other
        self.targeted = targeted
        self.reweight = reweight
        self.clf_all = clf_all

        if self.targeted:
            self.a_target = a_target
            self.a_other = a_other
        else:
            self.alpha_pdf = uniform.Uniform(
                low=torch.tensor(self.uni_low),
                high=torch.tensor(self.uni_high),
            )

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x: (torch.tensor): [B, L, D]
            target: (torch.tensor): [B, ]
            attn: Optional[torch.tensor]: [B, L] the attention weights to
                reweight the tokens in the sequence
        """
        if self.training:
            bsz, maxlen, d = x.size(0), x.size(1), x.size(2)
            new_bsz = self.n_samples
            # get attentions
            if self.reweight == "uniform":
                # uniform attention
                attn = torch.ones((bsz, maxlen)).to(x.device) / maxlen
            elif self.reweight == "gap":
                # calculate average pooling --> (B, L)
                # average per dimension: (B,L,D) -> (B,L)
                attn = torch.sum(x, dim=2) / d
                # 1e-5 avoids zero valued attention
                attn = attn.clamp(min=0.0) + 1e-5
                # norm value across B, sum over timesteps
                norm_val = torch.sum(attn, dim=1)
                # normalize, attn: (B,L)
                attn = attn / (norm_val.unsqueeze(1))
            else:
                pass  # attn already given in proper form (B, L)

            if self.targeted:
                # create diagonal
                alphas = (torch.eye(bsz).to(x.device)) \
                    * (self.a_target - self.a_other)
                # expand alphas
                alphas = alphas.repeat(new_bsz//bsz + 1, 1)[:new_bsz]
                # add self.a_other everywhere
                alphas += self.a_other
                # permute to get random places for a_target in the matrix
                perm = torch.randperm(new_bsz).to(x.device)
                alphas = alphas[perm]
                # sample dirichlet, already in (new_bsz, bsz) format
                mixing_factors = dirichlet.Dirichlet(alphas).sample()
                # expand to timesteps (N,B) --> N,L,B
                x_new.unsqueeze_(1)
                # x_new = torch.matmul(mixing_factors, x)
                # target_new = torch.matmul(mixing_factors, target)
            else:

                # sample alpha's for Dirichlet, one for each example in the B
                alphas = self.alpha_pdf.sample((bsz, )).to(x.device)
                # (N*L, B) --> since i want N L-tuples()
                # assumes per timestep normalization
                mixing_factors = \
                    dirichlet.Dirichlet(alphas).sample((new_bsz * maxlen))
                # (N*L, B) --> (N, L, B)
                mixing_factors = mixing_factors.reshape(new_bsz, maxlen, bsz)
            # re-weight step
            # hadamard (N,L,B)*(B,L)^T --> (N,L,B)
            # multiplies across the batch and timestep dim with attn weight
            mixing_factors = mixing_factors * attn.T
            # re-norm step across the batch dimension
            norm_factor = 1 / (torch.sum(mixing_factors, dim=2) + 1e-5)
            mixing_factors = mixing_factors * norm_factor.unsqueeze_(2)
            # custom matrix multiplication
            x_new = torch.einsum('nlb,bl->nl', mixing_factors, x)
            if self.clf_all:
                # one label for every token
                # (N, L)
                target_new = \
                    torch.einsum('nlb,bl->nl', mixing_factors, target)
            else:
                # compute the average over the sequence
                # (N,)
                target_new = \
                    torch.einsum('nlb,bl->n', mixing_factors, target) / maxlen
            x = x_new
            target = target_new
        return x, target


class ICM3xUp(nn.Module):
    def __init__(
            self,
            a_target_low: float = 1.0,
            a_target_high: float = 3.0,
            a_other_low: float = 0.1,
            a_other_high: float = 0.3,
            target_m: int = 0,
            n_samples: Optional[Union[int, None]] = None,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01,
            reweight: Optional[str] = "uniform",
    ):
        """Intra-Sample Cross-Modal Multimodal MixUp extension. (IC-M3xUp)
        Assumes a target modality and mixes all three modalities to replace
        this target modality. The intra flag handles the case of intra sample
        mixing or cross sample mixing. The mixing occurs in a targeted manner.
        Assumes 3 modalities.

        Args:
            a_target (float): the first alpha (l2r) in the beta distribution
            a_other (float): the 2nd/3rd alpha (l2r) in the beta distribution
            target_m (int): 0->L, 1->A, 2->V
            n_samples (int): more than one mixes in a particular modality
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
            reweight (str): add label reweighting based on each vector
                contribution
        """
        super(ICM3xUp, self).__init__()
        self.target_m = target_m
        self.add_noise = add_noise
        self.std = std
        self.reweight = reweight
        # get uniform parameters
        self.a_target_low = a_target_low
        self.a_target_high = a_target_high
        self.a_other_low = a_other_low
        self.a_other_high = a_other_high
        self.a_target = uniform.Uniform(self.a_target_low, self.a_target_high)

        if self.a_other_high == self.a_other_low:
            self.a_other = torch.tensor(self.a_other_low)
        else:
            self.a_other = uniform.Uniform(self.a_other_low, self.a_other_high)

        # alphas = torch.tensor([
        #     self.a_target,
        #     self.a_other,
        #     self.a_other
        # ])
        # # Assumes L,A,V order
        # self.alphas = torch.roll(alphas, self.target_m)

        self.n_samples = None
        if n_samples is not None:
            self.n_samples = n_samples

        # define the distribution from which we will sample lambda's
        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def get_mix_bsz(self, bsz):
        if self.n_samples is None or self.n_samples < bsz:
            return bsz
        else:
            return self.n_samples

    def get_attn(self, attn, bsz, d, n_mods, device):
        if attn is not None:
            pass
        else:
            if self.reweight == "uniform":
                    attn = torch.ones((bsz, n_mods)).to(device) / n_mods
            elif self.reweight == "gap":
                attn = []
                for k in range(n_mods):
                    # attn: (B)
                    attn.append(torch.sum(x[k], dim=1) / d)
                    # remove zero entries
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                # attn: (B,3)
                attn = torch.stack(attn, dim=1)
                # norm_per_modal: (B,)
                # FIXME division here must be buggy
                # norm_per_modal = torch.sum(attn, dim=1) / n_mods
                norm_per_modal = torch.sum(attn, dim=1)
                # normalize
                attn = attn / norm_per_modal.unsqueeze_(1)
            else:
                raise ValueError("Not a proper attn is given")
        return attn


    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # each tensor is [B, D]
            m_tar = x[self.target_m]
            n_mods = len(x)
            # l, a, v = x[0], x[1], x[2]
            bsz, d = m_tar.size(0), m_tar.size(1)

            attn = self.get_attn(attn, bsz, d, n_mods, m_tar.device)

            # get new_bsz
            new_bsz = self.get_mix_bsz(bsz)
            for k in range(n_mods):
                x[k] = x[k].repeat(new_bsz//bsz + 1, 1)[:new_bsz]

            # expand alphas
            alphas = torch.stack(
                [
                    self.a_target.sample((new_bsz,)).to(m_tar.device),
                    self.a_other.repeat(new_bsz).to(m_tar.device),
                    self.a_other.repeat(new_bsz).to(m_tar.device),
                ], dim=1
            )
            # alphas: (B, 3)
            alphas = torch.roll(alphas, self.target_m, 1)

            # expand attentions (B, 3) --> (N, 3)
            attn = attn.repeat(new_bsz // bsz + 1, 1)[:new_bsz]
            # sample dirichlet, already in (N, 3) format
            mixing_factors = dirichlet.Dirichlet(alphas).sample()
            # print(f"Mixing factors are {mixing_factors}")

            # combine and normalize mixing_factors
            mixing_factors = mixing_factors * attn
            # renormalize per row
            norm_factors = torch.sum(mixing_factors, dim=1)
            mixing_factors = mixing_factors / norm_factors.unsqueeze_(1)
            # print(f"Mixing factors are {torch.sum(mixing_factors, dim=0)/1000}")
            # print(mixing_factors.shape)
            for k in range(n_mods):
                if k == 0:
                    # (N,L)*(N)
                    x_new = x[k] * mixing_factors[:, k].unsqueeze(1)
                    # print(x_new.shape)
                else:
                    x_new = x_new + \
                        x[k] * mixing_factors[:, k].unsqueeze(1)
            if self.add_noise:
                x_new = x_new + \
                    self.additive_noise.sample(new_bsz, d).to(m_tar.device)
            # permute to get random places for a_target in the matrix
            if new_bsz > bsz:
                perm = torch.randperm(new_bsz).to(m_tar.device)
            # print(f"permutation size is {perm.shape}")
            # print(x_new.shape)
            if new_bsz > bsz:
                x_new = x_new[perm]
            # print(f"x_new shape is {x_new.shape}")
            for k in range(n_mods):
                if k == self.target_m:
                    x[k] = x_new
                else:
                    if new_bsz > bsz:
                        x[k] = x[k][perm]
            # fix target accordingly (B, ) --> (B,1 )
            # Comment: since i have assumed common label in both
            if new_bsz > bsz:
                target.unsqueeze_(1)
                target = target.repeat(new_bsz//bsz+1, 1)[:new_bsz]
                target = target[perm].reshape(-1)

        return x, target


class CIM3xUp(nn.Module):
    def __init__(
            self,
            a_target_low: float = 1.5,
            a_target_high: float = 2.0,
            a_other_low: float = 0.1,
            a_other_high: float = 0.1,
            target_m: Optional[Union[int, None]] = None,
            n_samples: Optional[Union[int, None]] = None,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01,
            reweight: Optional[str] = "uniform",
            keep_target_label: Optional[bool] = False,
            iid_mixes: Optional[bool] = False,
            override_a_other: Optional[bool] = False,
    ):
        """Cross Sample Intra Modal Multimodal MixUp extension. (CI-M3xUp)
        Assumes a pair of multimodal examples and mixes the two pais across
        each modality independently, therefore producing a new multimodal
        example. T1 is being mixed w T2, A1 is being mixed with A2 etc
        In all these implementations we assume that each modality contributes
        uniformly, i.e., the aggregated label is being weighted with 0.333
        Args:
            a_target (float): the first alpha (l2r) in the beta distribution
            a_other (float): the 2nd/3rd alpha (l2r) in the beta distribution
            target_m (int): specifier for a target modality 0->L, 1->A, 2->V
                Default is None
            n_samples (int): produce n_samples from different pair-mixes. This
                is not convex mixup but lends the idea of mulitple mixes.
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
            reweight (str): add label reweighting based on each vector
                contribution
            keep_target_label (bool): when True assumes targeted mixing but
                keeps the mulitmodal label same as the target
            iid_mixes (bool): when True resamples a different lambda
                for each modality (yet still targeted). Default: False
            override_a_other (bool): when True overrides the a_other argument
                with a single a_target. Set True in the case where you want to
                use a single a for all three modalities and mix them in
                arbitrary ways. Default: False.
        """
        super(CIM3xUp, self).__init__()
        self.add_noise = add_noise
        self.std = std
        self.reweight = reweight
        self.target_m = target_m  # target modality for mixing
        # import pdb; pdb.set_trace()
        if self.target_m == 'None':
            print("MIXING ALL MODALITIES")
            self.target_m = None
        self.keep_target_label = keep_target_label  # keeps target label
        self.iid_mixes = iid_mixes  # when true resamples beta per modality
        # TODO: add this functionality
        self.override_a_other = override_a_other
        if self.override_a_other:
            print(f"------------ functionality not implemented yet -------------------")

        # does not require self.targeted flag since directly affecting this
        # Beta here steers the algorithm accordingly
        self.a_target_low = a_target_low
        self.a_target_high = a_target_high
        self.a_other_low = a_other_low
        self.a_other_high = a_other_high
        if self.a_other_high == self.a_other_low:
            self.a_other = torch.tensor(self.a_other_low)
        self.a_target = uniform.Uniform(self.a_target_low, self.a_target_high)
        # self.beta = beta.Beta(
        #     torch.tensor(self.a_target),
        #     torch.tensor(self.a_other)
        #     )

        self.n_samples = n_samples
        if n_samples is not None:
            self.n_samples = n_samples

        # add noise to the lambdas
        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def get_mix_bsz(self, bsz):
        if self.n_samples is None:
            return bsz
        else:
            return self.n_samples

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # each tensor is [B, D]
            if self.target_m is None:
                m_tar = x[0]
            else:
                m_tar = x[self.target_m]
            n_mods = len(x)
            # l, a, v = x[0], x[1], x[2]
            bsz, d = m_tar.size(0), m_tar.size(1)

            if self.reweight == "uniform":
                attn = torch.ones((bsz, n_mods)).to(m_tar.device) / n_mods
                # print(f"atthention shape is {attn.shape}")
            elif self.reweight == "gap":
                attn = []
                for k in range(n_mods):
                    # attn: (B)
                    attn.append(torch.sum(x[k], dim=1) / d)
                    # remove zero entries
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                # attn: (B,3)
                attn = torch.stack(attn, dim=1)
                # norm_per_modal: (B,)
                # norm_per_modal = torch.sum(attn, dim=1) / n_mods
                norm_per_modal = torch.sum(attn, dim=1)
                # normalize
                attn = attn / norm_per_modal.unsqueeze(1)
            else:
                # already given attention
                attn

            # get new_bsz
            new_bsz = self.get_mix_bsz(bsz)
            if new_bsz > bsz:
                for k in range(n_mods):
                    x[k] = x[k].repeat(new_bsz//bsz + 1, 1)[:new_bsz]

            # permute to get random places for a_target in the matrix
            if new_bsz > bsz:
                perm = torch.randperm(new_bsz).to(m_tar.device)
                # expand attentions (B, 3) --> (N, 3)
                attn = attn.repeat(new_bsz // bsz + 1, 1)[:new_bsz]
                # reshape is required when target is a (B,) tensor
                target = \
                    target.repeat(new_bsz // bsz + 1, 1).reshape(-1)[:new_bsz]
                # print(f"new target shape is {target}")
            else:
                # here we need randperm to mix random examples in the batch
                perm = torch.randperm(bsz).to(m_tar.device)

            # print(f"The permutation I made is {perm}")

            # clone and permute to create target batch
            x_perm = [x[k][perm] for k in range(n_mods)]
            attn_perm = attn[perm]
            target_perm = target[perm]

            # sample beta to get mixing_factors
            #  mf: List [(new_bsz, 2)]
            mixing_factors = []
            c1 = self.a_target.sample((new_bsz, )).to(m_tar.device)
            beta_pdf = beta.Beta(
                c1, self.a_other.to(m_tar.device)
            )
            if self.iid_mixes:
                for _ in range(n_mods):
                    # tmp_factors = self.beta.sample((new_bsz,)).to(m_tar.device)
                    tmp_factors = beta_pdf.sample()
                    mixing_factors.append(
                        torch.stack([tmp_factors, 1-tmp_factors], dim=1)
                    )
            else:
                tmp_factors = self.beta.sample()
                mixing_factors.append(
                        torch.stack([tmp_factors, 1-tmp_factors], dim=1)
                    )
                for _ in range(n_mods-1):
                    mixing_factors.append(mixing_factors[0])

            # (B, 6) --> each pair of cols is the lam and 1-lam for each
            # modality, i.e, the first two text and so on....
            # these values are not normalized
            new_target = []
            for k in range(n_mods):
                # print(f"Original tensor is {x[k]}")
                # print(f"Shuffled tensor is {x[k]}")
                # tmp_factor: (B, 2)
                with torch.no_grad():
                    tmp_factor = mixing_factors[k]
                    # normalize, tmp_factor:(B, 2), attn:(B, n_mods
                    lam_0 = tmp_factor[:, 0].clone()
                    lam_1 = tmp_factor[:, 1].clone()
                    lam_0 = lam_0 * attn[:, k]
                    lam_1 = lam_1 * attn_perm[:, k]
                    tmp_factor[:, 0] = lam_0
                    tmp_factor[:, 1] = lam_1
                    # tmp_factor[:, 0] = lam_0 * attn[:, k]
                    # tmp_factor[:, 1] = lam_1 * attn_perm[:, k]
                    # # tmp_factor[:, 0] = lam_0 * attn[:, k]
                    # tmp_factor[:, 1] = lam_1 * attn_perm[:, k]
                    # print(tmp_factor.size())
                    tmp_factor = tmp_factor / \
                        torch.sum(tmp_factor, dim=1).unsqueeze(1)
                    # store back to List of (B, 2)
                    mixing_factors[k] = tmp_factor
                    # print(f"Mixing factors are {mixing_factors[k]}")
                    # get mixed representations and overwrite them on x_perm
                x_perm[k] = \
                    x[k] * tmp_factor[:, 0].unsqueeze(1) \
                    + x_perm[k] * tmp_factor[:, 1].unsqueeze(1)
                if self.add_noise:
                    x_perm[k] = x_perm[k] + \
                        self.additive_noise.sample(new_bsz, d).to(m_tar.device)
                # overwrite on target_perm
                new_target.append(
                    target * tmp_factor[:, 0] + target_perm * tmp_factor[:, 1]
                )
            # import pdb; pdb.set_trace()

            # if target modality exists
            if self.target_m is not None:
                x[self.target_m] = x_perm[self.target_m]
                target = \
                    ((n_mods-1)/n_mods) * target \
                    + (1/n_mods) * new_target[self.target_m]
            elif self.keep_target_label:
                for k in range(n_mods):
                    # .clone() retains backward()
                    x[k] = x_perm[k].clone()
                # target remains unaffected
            else:
                for k in range(n_mods):
                    # .clone() retains backward()
                    x[k] = x_perm[k]
                    # average target of three modalities
                    # if k == 0:
                    #     target = new_target[0] * 0.333
                    # else:
                    #     target += new_target[k] * 0.333
                target = torch.mean(torch.stack(new_target, dim=1), dim=1)
                # import pdb; pdb.set_trace()
            # add a final permutation since x, target are stored
            # as copies of the initial
            if new_bsz > bsz:
                # add an extra permutation to avoid patterns in the output
                # a perm already exists --> use this
                for k in range(n_mods):
                    x[k] = x[k][perm]
                target = target[perm]

        return x, target


class M2MixUp(nn.Module):
    def __init__(
            self,
            a : float = 1.0,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01,
    ):
        """
        Cross Sample Intra Modal MixUp extension. (M2MixUp)
        Assumes a pair of multimodal examples and mixes the two pais across
        each modality independently, therefore producing a new multimodal
        example. T1 is being mixed w T2, A1 is being mixed with A2 etc
        In all these implementations we assume that each modality contributes
        uniformly, i.e., the aggregated label is being weighted with 0.333
        Args:
            a(float): the concentration of the beta distribution
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
        """
        super(M2MixUp, self).__init__()
        self.add_noise = add_noise
        self.std = std
        self.perm_flag = False # when False splits the batch to half
        self.iid_mixes = False # when False splits the batch to half
        # Beta steers the algorithm accordingly
        self.a = a
        self.beta = beta.Beta(
            torch.tensor(self.a),
            torch.tensor(self.a)
            )

        # add noise to the lambdas
        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def forward(self, x, target):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
        """
        if self.training:
            # each tensor is [B, D]
            n_mods = len(x)
            # l, a, v = x[0], x[1], x[2]
            bsz = x[0].size(0)

            # sample beta to get mixing_factors
            #  mf: List [(bsz//2, 2)]
            mixing_factors = []
            tmp_factors = self.beta.sample((bsz//2, )).to(x[0].device)
            mixing_factors.append(
                    torch.stack([tmp_factors, 1-tmp_factors], dim=1)
                )
            for _ in range(n_mods-1):
                mixing_factors.append(mixing_factors[0])

            y_mix = []
            x_mix = []
            for k in range(n_mods):
                x0 = x[k][:bsz//2, ...]
                x1 = x[k][bsz//2:, ...]
                # multimodal target shared across modalities
                y0 = target[:bsz//2, ...]
                y1 = target[bsz//2:, ...]
                l0 = mixing_factors[k][:, 0]
                l1 = mixing_factors[k][:, 1]
                x_mix.append(x0 * l0.unsqueeze(1) + x1 * l1.unsqueeze(1))
                y_mix.append(y0 * l0 + y1 * l1)
            # for k in range(n_mods):
            #     # .clone() retains backward()
            #     x[k] = x_mix[k]
            x = [tensor.clone() for tensor in x_mix]
            target = torch.mean(torch.stack(y_mix, dim=1), dim=1)
            # import pdb; pdb.set_trace()
        return x, target


class ConvexCIM3xUp(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_target_low: float = 0.5,
            uni_target_high: Optional[float] = 2.0,
            uni_other_low: float = 0.5,
            uni_other_high: Optional[float] = 2.0,
            targeted: Optional[bool] = False,
            a_target: Optional[float] = 3.0,
            a_other: Optional[float] = 0.1,
            target_m: Optional[Union[int, None]] = None,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01,
            reweight: Optional[str] = "uniform",
            iid_mixes: Optional[bool] = False,
    ):
        """Convex Cross Sample Intra Modal Multimodal MixUp extension.
        (Convex CI-M3xUp) Assumes CONVEX multimodal examples and mixes the
        tuples across each modality independently, therefore producing N new
        multimodal examples. T1 is being mixed w T2,3..., A1 is being mixed 
        with A2,3,4,... etc. In all these implementations we assume that each 
        modality contributes uniformly, i.e., the aggregated label is being 
        weighted with 0.333
        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
            targeted (bool): by default does not use the targeted approach
                where a target vector is held and the others are interpolated
                via small factors
            a_target (float): the concentration of the target sample, 3,4,5
            a_other (float): the concentration of the other sample, 0.1, 0.3
            a_target (float): the first alpha (l2r) in the beta distribution
            a_other (float): the 2nd/3rd alpha (l2r) in the beta distribution
            target_m (int): specifier for a target modality 0->L, 1->A, 2->V
                Default is None
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
            reweight (str): add label reweighting based on each vector
                contribution
            iid_mixes (bool): when True resamples a different lambda
                for each modality (yet still targeted). Default: False
        """
        super(ConvexCIM3xUp, self).__init__()
        # self.uni_low = uni_low
        # self.uni_high = uni_high
        self.uni_target_low = uni_target_low
        self.uni_target_high = uni_target_high
        self.uni_other_low = uni_other_low
        self.uni_other_high = uni_other_high
        self.targeted = targeted  # if True assumes a "main" target
        self.a_target = a_target
        self.a_other = a_other
        self.add_noise = add_noise
        self.std = std
        self.reweight = reweight
        self.target_m = target_m  # target modality for mixing
        self.iid_mixes = iid_mixes  # when true resamples beta per modality

        if self.target_m == 'None':
            print("MIXING ALL MODALITIES")
            self.target_m = None

        # import pdb; pdb.set_trace()
        # does not require self.targeted flag since directly affecting this
        # Beta here steers the algorithm accordingly
        if self.targeted:
            print("Onogoing with targeted scenario with: ")
            print(f"OTHER: U[{self.uni_other_low}, {self.uni_other_high}]")
            print(f"TARGET: U[{self.uni_target_low}, {self.uni_target_low}]")
            self.alpha_high = uniform.Uniform(
                low=torch.tensor(self.uni_target_low, device='cuda'),
                high=torch.tensor(self.uni_target_low + 0.001, device='cuda'),
            )
            self.alpha_low = uniform.Uniform(
                low=torch.tensor(self.uni_other_low, device='cuda'),
                high=torch.tensor(self.uni_other_high, device='cuda'),
                # high=torch.tensor(self.uni_other_high),
            )
        else:
            print(f"Ongoing with untargeted scenario with low={self.uni_other_low} "
                  f"high={self.uni_other_high}")
            self.alpha_pdf = uniform.Uniform(
                low=torch.tensor(self.uni_other_low, device='cuda'),
                high=torch.tensor(self.uni_other_high, device='cuda'),
            )

        self.n_samples = n_samples
        if n_samples is not None:
            print(f"Proceeding with generating {n_samples} per mini-batch")
            self.n_samples = n_samples

        # add noise to the lambdas
        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def get_mix_bsz(self, bsz):
        if self.n_samples is None:
            return bsz
        else:
            return self.n_samples

    def get_attn(self, x, attn, bsz, d, n_mods):
        if attn is None:
            if self.reweight == "uniform":
                attn = torch.ones((bsz, n_mods)).to(x[0].device) / n_mods
            elif self.reweight == "gap":
                # print("Onoging with GAP pooling")
                attn = []
                for k in range(n_mods):
                    # attn: (B)
                    attn.append(torch.sum(x[k], dim=1) / d)
                    # remove zero entries
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                # attn: (B,3)
                attn = torch.stack(attn, dim=1)
                # norm_per_modal: (B,)
                norm_per_modal = torch.sum(attn, dim=1)
                # normalize
                attn = attn / norm_per_modal.unsqueeze_(1)
            else:
                raise ValueError("Not a valid attention calculation")
        return attn

    def get_mix_factors(self, bsz, n_mods, device):
        if self.targeted:
            # print("spot - 0")
            # create diagonal
            a_target = self.alpha_high.sample((bsz, )) #.to(device)
            a_other = self.alpha_low.sample((bsz, )) #.to(device)
            alphas = (torch.eye(bsz).to(device)) * (a_target - a_other)
            # alphas = (torch.eye(bsz).to(device)) \
            #     * (self.a_target - self.a_other)
            # expand alphas
            alphas = alphas.repeat(self.n_samples//bsz + 1, 1)[:self.n_samples]
            # add self.a_other everywhere
            alphas += self.a_other
            # permute to get random places for a_target in the matrix
            perm = torch.randperm(self.n_samples).to(device)
            alphas = alphas[perm]
            # sample dirichlet, already in (N, B) format
            # mixing_factors = dirichlet.Dirichlet(alphas).sample()
            mixing_factors_pdf = dirichlet.Dirichlet(alphas)
            # x_new = torch.matmul(mixing_factors, x)
            # target_new = torch.matmul(mixing_factors, target)

            # handle single target modality case
            if self.target_m is not None:
                one_hot_mix = torch.eye(bsz).to(device)
                one_hot_mix = \
                    one_hot_mix.repeat(self.n_samples//bsz+1, 1)[:self.n_samples]
                # permuted (N, B)
                one_hot_mix = one_hot_mix[perm]

                # generate mixing_factors for each modality
                lam = [one_hot_mix for _ in range(n_mods)]
                lam[self.target_m] = mixing_factors_pdf.sample()
            else:  # no target modality scenario
                if self.iid_mixes:
                    lam = [mixing_factors_pdf.sample() for _ in range(n_mods)]
                else:
                    mixing_factors = mixing_factors_pdf.sample()
                    lam = [mixing_factors for _ in range(n_mods)]
        else:
            # print("spot - 1")
            # sample alpha's for Dirichlet
            alphas = self.alpha_pdf.sample((bsz, )).to(device)
            # print(alphas)
            # print(f"the shape of x is {x.shape}")
            # (new_bsz, bsz)
            mixing_factors_pdf = dirichlet.Dirichlet(alphas)
            # mixing_factors_pdf = \
            #     dirichlet.Dirichlet(alphas).sample((self.n_samples, ))

            # handle target modality scenario
            if self.target_m is not None:
                # (N, B)
                # here gets the target of the largest mixing_factor as a
                # one-hot approximation
                mixing_factors = mixing_factors_pdf.sample((self.n_samples, ))
                max_indices = torch.argmax(mixing_factors, dim=1)
                one_hot_mix = \
                    torch.eye(
                        mixing_factors.shape[1],
                        device=device
                    )[max_indices]

                lam = [one_hot_mix for _ in range(n_mods)]
                lam[self.target_m] = \
                    mixing_factors_pdf.sample((self.n_samples, ))
            else:
                # print("spot - 2 - untargeted")
                if self.iid_mixes:
                    # print("spot - 3 - aniso")
                    lam = [
                        mixing_factors_pdf.sample((self.n_samples, ))
                        for _ in range(n_mods)
                    ]
                else:
                    # print("spot - 4 - iso")
                    mixing_factors = \
                        mixing_factors_pdf.sample((self.n_samples, ))
                    lam = [mixing_factors for _ in range(n_mods)]

            # List[(N,B)]
        return lam

    @staticmethod
    def norm_mix_factors(mixing_factors, attn, n_mods):
        for k in range(n_mods):
            # reweight lambdas
            mixing_factors[k] = \
                mixing_factors[k] * attn[..., k].unsqueeze(0)
            # row normalization (N, B), summation over the B
            norm_factor = torch.sum(mixing_factors[k], dim=1)
            mixing_factors[k] = \
                mixing_factors[k] / norm_factor.unsqueeze_(1)
        return mixing_factors

    def convex_sums(self, mixing_factors, x, target, n_mods):
        """convex sum calculator function"""
        x_new = []
        y_new = []

        for k in range(n_mods):
            # (N,B) @ (B,D) --> (N, D)
            x_new.append(torch.matmul(mixing_factors[k], x[k]))
            # (N,B) @ (B,1) --> (N, 1)
            y_new.append(torch.matmul(
                mixing_factors[k], target.unsqueeze(1))
            )
        # aggregated label
        y_new = torch.sum(torch.stack(y_new, dim=1), dim=1) * (1/n_mods)
        y_new.squeeze_(1)  # (B,1) --> (B,)
        return x_new, y_new

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # each tensor is [B, D], get necessary dimensions
            x_m = x[0]
            n_mods = len(x)
            bsz, d = x_m.size(0), x_m.size(1)

            with torch.no_grad():
                # calculate attentions, attn (B, 3)
                attn = self.get_attn(x, attn, bsz, d, n_mods)
                # print(f"Attnetions are {attn}")

                # calculate mixing factors, [(N, B), (N, B), (N, B)]
                mixing_factors = self.get_mix_factors(bsz, n_mods, x_m.device)

                # normalization step
                mixing_factors = self.norm_mix_factors(
                    mixing_factors, attn, n_mods
                )
                # print(f"Mixing factors are {mixing_factors}")

            # calculate convex sums
            x, target = self.convex_sums(mixing_factors, x, target, n_mods)

        return x, target


class M3ltiMix(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_low: Optional[float] = 0.5,
            uni_high: Optional[float] = 2.0,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01,
            reweight: Optional[str] = "uniform",
            iid_mixes: Optional[bool] = False,
    ):
        """Convex Cross Sample Intra Modal MultiMix extension, aka, M3ltiMix
        (Convex CI-M3xUp) Assumes CONVEX multimodal examples and mixes the
        tuples across each modality independently, therefore producing N new
        multimodal examples. T1 is being mixed w T2,3..., A1 is being mixed 
        with A2,3,4,... etc. In all these implementations we assume that each 
        modality contributes uniformly, i.e., the aggregated label is being 
        weighted with 0.333
        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
            reweight (str): add label reweighting based on each vector
                contribution
            iid_mixes (bool): when True resamples a different lambda
                for each modality (yet still targeted). Default: False
        """
        super(M3ltiMix, self).__init__()
        self.uni_low = uni_low
        self.uni_high = uni_high
        self.add_noise = add_noise
        self.std = std
        self.reweight = reweight
        self.iid_mixes = iid_mixes  # when true resamples beta per modality

        print(f"Ongoing with with low={self.uni_low} "
              f"high={self.uni_high}")
        self.alpha_pdf = uniform.Uniform(
            low=torch.tensor(self.uni_low, device='cuda'),
            high=torch.tensor(self.uni_high, device='cuda'),
        )

        self.n_samples = n_samples
        if n_samples is not None:
            print(f"Proceeding with generating {n_samples} per mini-batch")
            self.n_samples = n_samples

        # add noise to the lambdas
        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def get_mix_bsz(self, bsz):
        if self.n_samples is None:
            return bsz
        else:
            return self.n_samples

    def get_attn(self, x, attn, bsz, d, n_mods):
        if attn is None:
            if self.reweight == "uniform":
                attn = torch.ones((bsz, n_mods)).to(x[0].device) / n_mods
            elif self.reweight == "gap":
                # print("Onoging with GAP pooling")
                attn = []
                for k in range(n_mods):
                    # attn: (B)
                    attn.append(torch.sum(x[k], dim=1) / d)
                    # remove zero entries
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                # attn: (B,3)
                attn = torch.stack(attn, dim=1)
                # norm_per_modal: (B,)
                norm_per_modal = torch.sum(attn, dim=1)
                # normalize
                attn = attn / norm_per_modal.unsqueeze_(1)
            else:
                raise ValueError("Not a valid attention calculation")
        return attn

    def get_mix_factors(self, bsz, n_mods, device):
        # sample alpha's for Dirichlet
        alphas = self.alpha_pdf.sample((bsz, )).to(device)
        # (new_bsz, bsz)
        mixing_factors_pdf = dirichlet.Dirichlet(alphas)

        if self.iid_mixes:
            # anisotropic mixing
            lam = [
                mixing_factors_pdf.sample((self.n_samples, ))
                for _ in range(n_mods)
            ]
        else:
            # isotropic mixing
            mixing_factors = \
                mixing_factors_pdf.sample((self.n_samples, ))
            lam = [mixing_factors for _ in range(n_mods)]

        # List[(N,B)]
        return lam

    @staticmethod
    def norm_mix_factors(mixing_factors, attn, n_mods):
        for k in range(n_mods):
            # reweight lambdas
            mixing_factors[k] = \
                mixing_factors[k] * attn[..., k].unsqueeze(0)
            # row normalization (N, B), summation over the B
            norm_factor = torch.sum(mixing_factors[k], dim=1)
            mixing_factors[k] = \
                mixing_factors[k] / norm_factor.unsqueeze_(1)
        return mixing_factors

    def convex_sums(self, mixing_factors, x, target, n_mods):
        """convex sum calculator function"""
        x_new = []
        y_new = []

        for k in range(n_mods):
            # (N,B) @ (B,D) --> (N, D)
            x_new.append(torch.matmul(mixing_factors[k], x[k]))
            # (N,B) @ (B,1) --> (N, 1)
            y_new.append(torch.matmul(
                mixing_factors[k], target.unsqueeze(1))
            )
        # aggregated label
        y_new = torch.sum(torch.stack(y_new, dim=1), dim=1) * (1/n_mods)
        y_new.squeeze_(1)  # (B,1) --> (B,)
        return x_new, y_new

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # each tensor is [B, D], get necessary dimensions
            x_m = x[0]
            n_mods = len(x)
            bsz, d = x_m.size(0), x_m.size(1)

            with torch.no_grad():
                # calculate attentions, attn (B, 3)
                attn = self.get_attn(x, attn, bsz, d, n_mods)
                # print(f"Attnetions are {attn}")

                # calculate mixing factors, [(N, B), (N, B), (N, B)]
                mixing_factors = self.get_mix_factors(bsz, n_mods, x_m.device)

                # normalization step
                mixing_factors = self.norm_mix_factors(
                    mixing_factors, attn, n_mods
                )
                # print(f"Mixing factors are {mixing_factors}")

            # calculate convex sums
            x, target = self.convex_sums(mixing_factors, x, target, n_mods)

        return x, target


class MskCvxCIM3xUp(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_low: float = 0.5,
            uni_high: float = 1.5,
            mask_low: float = 0.5,
            mask_high: float = 1.0,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01,
            reweight: Optional[str] = "uniform",
            iid_mixes: Optional[bool] = False,
            iso_mask: Optional[bool] = True,
    ):
        """Convex Cross Sample Intra Modal Multimodal MixUp extension.
        (Convex CI-M3xUp) Assumes CONVEX multimodal examples and mixes the
        tuples across each modality independently, therefore producing N new
        multimodal examples. T1 is being mixed w T2,3..., A1 is being mixed 
        with A2,3,4,... etc. In all these implementations we assume that each 
        modality contributes uniformly, i.e., the aggregated label is being 
        weighted with 0.333
        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
            targeted (bool): by default does not use the targeted approach
                where a target vector is held and the others are interpolated
                via small factors
            a_target (float): the concentration of the target sample, 3,4,5
            a_other (float): the concentration of the other sample, 0.1, 0.3
            a_target (float): the first alpha (l2r) in the beta distribution
            a_other (float): the 2nd/3rd alpha (l2r) in the beta distribution
            target_m (int): specifier for a target modality 0->L, 1->A, 2->V
                Default is None
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
            reweight (str): add label reweighting based on each vector
                contribution
            iid_mixes (bool): when True resamples a different lambda
                for each modality (aniso) Default: False (iso)
            iso_mask (bool): masks the same sample across all modalities
        """
        super(MskCvxCIM3xUp, self).__init__()
        self.uni_low = uni_low
        self.uni_high = uni_high
        self.mask_low = mask_low
        self.mask_high = mask_high
        self.add_noise = add_noise
        self.std = std
        self.reweight = reweight
        self.iid_mixes = iid_mixes  # when true resamples beta per modality
        self.iso_mask = iso_mask

        print("MIXING ALL MODALITIES")

        print(f"Ongoing with untargeted scenario with" 
              f"low={self.uni_low} "
              f"high={self.uni_high}")

        self.alpha_pdf = uniform.Uniform(
            low=torch.tensor(self.uni_low, device='cuda'),
            high=torch.tensor(self.uni_high, device='cuda'),
        )

        self.mask_pdf = uniform.Uniform(
            low=torch.tensor(self.mask_low, device='cuda'),
            high=torch.tensor(self.mask_high, device='cuda'),
        )

        self.n_samples = n_samples
        if n_samples is not None:
            print(f"Proceeding with generating {n_samples} per mini-batch")
            self.n_samples = n_samples

        # add noise to the lambdas
        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def get_mix_bsz(self, bsz):
        if self.n_samples is None:
            return bsz
        else:
            return self.n_samples

    def get_attn(self, x, attn, bsz, d, n_mods):
        if attn is None:
            if self.reweight == "uniform":
                attn = torch.ones((bsz, n_mods)).to(x[0].device) / n_mods
            elif self.reweight == "gap":
                # print("Onoging with GAP pooling")
                attn = []
                for k in range(n_mods):
                    # attn: (B)
                    attn.append(torch.sum(x[k], dim=1) / d)
                    # remove zero entries
                    # attn[k] = attn[k].clamp(min=0.0) + 1e-5
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                # attn: (B,3)
                attn = torch.stack(attn, dim=1)
                # norm_per_modal: (B,)
                norm_per_modal = torch.sum(attn, dim=1)
                # normalize
                attn = attn / norm_per_modal.unsqueeze_(1)
            else:
                raise ValueError("Not a valid attention calculation")
        return attn

    def mlabel_get_attn(self, x, attn, bsz, n_mods):
        if attn is None:
            if self.reweight == "uniform":
                attn = torch.ones((bsz, n_mods)).to(x[0].device) / n_mods
            elif self.reweight == "gap":
                # print("Onoging with GAP pooling")
                attn = []
                for k in range(n_mods):
                    d = x[k].size(1)
                    # attn: (B)
                    attn.append(torch.sum(x[k], dim=1) / d)
                    # remove zero entries
                    # attn[k] = attn[k].clamp(min=0.0) + 1e-5
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                # attn: (B,3)
                attn = torch.stack(attn, dim=1)
                # norm_per_modal: (B,)
                norm_per_modal = torch.sum(attn, dim=1)
                # normalize
                attn = attn / norm_per_modal.unsqueeze_(1)
            else:
                raise ValueError("Not a valid attention calculation")
        return attn

    @staticmethod
    def replace_zero_rows_with_vector(tensor, p):
        mask = (tensor.sum(dim=1) == 0).unsqueeze(1)
        num_zero_rows = mask.sum()
        vector_size = tensor.size(1)
        vector = \
            torch.bernoulli(
                torch.full((num_zero_rows, vector_size), p, device="cuda")
            )
        replaced_tensor = tensor.masked_scatter(mask, vector)
        return replaced_tensor

    @staticmethod
    def replace_zero_one_rows_with_vector(tensor, p):
        mask_one = (tensor.sum(dim=1) == 1).unsqueeze(1)
        mask_zero = (tensor.sum(dim=1) == 0).unsqueeze(1)
        mask = mask_one + mask_zero # logical AND
        num_zero_rows = mask.sum()
        vector_size = tensor.size(1)
        vector = \
            torch.bernoulli(
                torch.full((num_zero_rows, vector_size), p, device="cuda")
            )
        replaced_tensor = tensor.masked_scatter(mask, vector)
        return replaced_tensor

    def get_mix_factors(self, bsz, n_mods, device):
        # sample alpha's for Dirichlet
        alphas = self.alpha_pdf.sample((bsz, ))
        # alphas = alphas.repeat(1, bsz)
        # sample p_mask (p_keep) for Dirichlet
        p_keep = self.mask_pdf.sample((self.n_samples, bsz))
        # (new_bsz, bsz)
        mixing_factors_pdf = dirichlet.Dirichlet(alphas)
        mask_pdf = bernoulli.Bernoulli(p_keep)
        # mixing_factors_pdf = \
        #     dirichlet.Dirichlet(alphas).sample((self.n_samples, ))

        # print("spot - 2 - untargeted")
        if self.iid_mixes:
            # print("spot - 3 - aniso")
            mask = mask_pdf.sample()
            mask = self.replace_zero_one_rows_with_vector(
                mask, 0.5*(self.mask_low+self.mask_high)
            )
            # import pdb; pdb.set_trace()
            # TODO: better implementation when masking alphas
            lam = []
            for _ in range(n_mods):
                tmp_factors = mixing_factors_pdf.sample((self.n_samples, )) * mask
                norm_factor = torch.sum(tmp_factors, dim=1).unsqueeze_(1) + 1e-5 # smoothing
                lam.append(tmp_factors / norm_factor)
        else:
            # print("spot - 4 - iso")
            mask = mask_pdf.sample()
            mask = self.replace_zero_one_rows_with_vector(
                mask, 0.5*(self.mask_low+self.mask_high)
            )
            tmp_factors = \
                mixing_factors_pdf.sample((self.n_samples, )) * mask
            norm_factor = torch.sum(tmp_factors, dim=1).unsqueeze_(1) + 1e-5
            mixing_factors = tmp_factors / norm_factor
            lam = [mixing_factors for _ in range(n_mods)]

        # List[(N,B)]
        return lam

    @staticmethod
    def norm_mix_factors(mixing_factors, attn, n_mods):
        for k in range(n_mods):
            # reweight lambdas
            mixing_factors[k] = \
                mixing_factors[k] * attn[..., k].unsqueeze(0)
            # row normalization (N, B), summation over the B
            norm_factor = torch.sum(mixing_factors[k], dim=1) + 1e-5
            mixing_factors[k] = \
                mixing_factors[k] / norm_factor.unsqueeze_(1)
        return mixing_factors

    def convex_sums(self, mixing_factors, x, target, n_mods):
        """convex sum calculator function"""
        x_new = []
        y_new = []

        for k in range(n_mods):
            # (N,B) @ (B,D) --> (N, D)
            x_new.append(torch.matmul(mixing_factors[k], x[k]))
            # (N,B) @ (B,1) --> (N, 1)
            y_new.append(torch.matmul(
                mixing_factors[k], target.unsqueeze(1))
            )
        # aggregated label
        y_new = torch.sum(torch.stack(y_new, dim=1), dim=1) * (1/n_mods)
        y_new.squeeze_(1)  # (B,1) --> (B,)
        return x_new, y_new

    def mlabel_convex_sums(self, mixing_factors, x, target, n_mods):
        """multilabel convex sum calculator function"""
        x_new = []
        y_unimodal_new = []
        y_multimodal_new = []

        for k in range(n_mods):
            # (N,B) @ (B,D) --> (N, D)
            x_new.append(torch.matmul(mixing_factors[k], x[k]))
            # (N,B) @ (B,1) --> (N, 1)
            y_unimodal_new.append(torch.matmul(mixing_factors[k], target[k].unsqueeze(1)).squeeze_(1))
            y_multimodal_new.append(torch.matmul(mixing_factors[k], target[-1].unsqueeze(1)))
        # aggregated label
        y_multimodal_new = torch.sum(torch.stack(y_multimodal_new, dim=1), dim=1) * (1/n_mods)
        y_multimodal_new.squeeze_(1)  # (B,1) --> (B,)
        # collect both unimodal and multimodal to return
        y_unimodal_new.append(y_multimodal_new)
        return x_new, y_unimodal_new

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # print(f"Unilabel forward pass")
            # each tensor is [B, D], get necessary dimensions
            x_m = x[0]
            n_mods = len(x)
            bsz, d = x_m.size(0), x_m.size(1)

            with torch.no_grad():
                # calculate attentions, attn (B, 3)
                attn = self.get_attn(x, attn, bsz, d, n_mods)
                # print(f"Attnetions are {attn}")

                # calculate mixing factors, [(N, B), (N, B), (N, B)]
                mixing_factors = self.get_mix_factors(bsz, n_mods, x_m.device)

                # normalization step
                mixing_factors = self.norm_mix_factors(
                    mixing_factors, attn, n_mods
                )
                # print(f"Mixing factors are {mixing_factors}")

            # calculate convex sums
            x, target = self.convex_sums(mixing_factors, x, target, n_mods)

        return x, target

    def multilabel_forward(self, x, target, attn=None):
        """
        Multilabel version which assumes x: [text, audio, video]
        and y: [text, audio, video, multimodal]
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # print(f"Multilabel forward pass")
            # each tensor is [B, D], get necessary dimensions
            x_m = x[0]
            n_mods = len(x)
            bsz = x_m.size(0)

            with torch.no_grad():
                # calculate attentions, attn (B, 3)
                attn = self.mlabel_get_attn(x, attn, bsz, n_mods)

                # calculate mixing factors, [(N, B), (N, B), (N, B)]
                mixing_factors = self.get_mix_factors(bsz, n_mods, x_m.device)

                # normalization step
                mixing_factors = self.norm_mix_factors(
                    mixing_factors, attn, n_mods
                )
                # print(f"Mixing factors are {mixing_factors}")

            # calculate convex sums
            x, target = self.mlabel_convex_sums(mixing_factors, x, target, n_mods)

        return x, target

class M2PowMix(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_low: float = 0.5,
            uni_high: float = 1.5,
            mask_low: float = 0.5,
            mask_high: float = 1.0,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01,
            reweight: Optional[str] = "uniform",
            iid_mixes: Optional[bool] = False,
            iso_mask: Optional[bool] = True,
    ):
        """MultiModal version of PowerMixUp, a generic mixing framework. 
        (M2PowMix) Assumes interpolations on random subsets of the batch. 
        Mixes the tuples across each modality independently, therefore producing N new
        multimodal examples. T1 is being mixed w T2,3..., A1 is being mixed 
        with A2,3,4,... etc. In all these implementations we assume that each 
        modality contributes uniformly, i.e., the aggregated label is being 
        weighted with 0.333
        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
            targeted (bool): by default does not use the targeted approach
                where a target vector is held and the others are interpolated
                via small factors
            a_target (float): the concentration of the target sample, 3,4,5
            a_other (float): the concentration of the other sample, 0.1, 0.3
            a_target (float): the first alpha (l2r) in the beta distribution
            a_other (float): the 2nd/3rd alpha (l2r) in the beta distribution
            target_m (int): specifier for a target modality 0->L, 1->A, 2->V
                Default is None
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
            reweight (str): add label reweighting based on each vector
                contribution
            iid_mixes (bool): when True resamples a different lambda
                for each modality (aniso) Default: False (iso)
            iso_mask (bool): masks the same sample across all modalities
        """
        super(M2PowMix, self).__init__()
        self.uni_low = uni_low
        self.uni_high = uni_high
        self.mask_low = mask_low
        self.mask_high = mask_high
        self.add_noise = add_noise
        self.std = std
        self.reweight = reweight
        self.iid_mixes = iid_mixes  # when true resamples beta per modality
        self.iso_mask = iso_mask

        print("MIXING ALL MODALITIES")

        print(f"Ongoing with " 
              f"low={self.uni_low} "
              f"high={self.uni_high}")

        self.alpha_pdf = uniform.Uniform(
            low=torch.tensor(self.uni_low, device='cuda'),
            high=torch.tensor(self.uni_high, device='cuda'),
        )

        self.mask_pdf = uniform.Uniform(
            low=torch.tensor(self.mask_low, device='cuda'),
            high=torch.tensor(self.mask_high, device='cuda'),
        )

        self.n_samples = n_samples
        if n_samples is not None:
            print(f"Proceeding with generating {n_samples} per mini-batch")
            self.n_samples = n_samples

        # add noise to the lambdas
        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def get_mix_bsz(self, bsz):
        if self.n_samples is None:
            return bsz
        else:
            return self.n_samples

    def get_attn(self, x, attn, bsz, d, n_mods):
        if attn is None:
            if self.reweight == "uniform":
                attn = torch.ones((bsz, n_mods)).to(x[0].device) / n_mods
            elif self.reweight == "gap":
                # print("Onoging with GAP pooling")
                attn = []
                for k in range(n_mods):
                    # attn: (B)
                    attn.append(torch.sum(x[k], dim=1) / d)
                    # remove zero entries
                    # attn[k] = attn[k].clamp(min=0.0) + 1e-5
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                # attn: (B,3)
                attn = torch.stack(attn, dim=1)
                # norm_per_modal: (B,)
                norm_per_modal = torch.sum(attn, dim=1)
                # normalize
                attn = attn / norm_per_modal.unsqueeze_(1)
            else:
                raise ValueError("Not a valid attention calculation")
        return attn

    def mlabel_get_attn(self, x, attn, bsz, n_mods):
        if attn is None:
            if self.reweight == "uniform":
                attn = torch.ones((bsz, n_mods)).to(x[0].device) / n_mods
            elif self.reweight == "gap":
                # print("Onoging with GAP pooling")
                attn = []
                for k in range(n_mods):
                    d = x[k].size(1)
                    # attn: (B)
                    attn.append(torch.sum(x[k], dim=1) / d)
                    # remove zero entries
                    # attn[k] = attn[k].clamp(min=0.0) + 1e-5
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                # attn: (B,3)
                attn = torch.stack(attn, dim=1)
                # norm_per_modal: (B,)
                norm_per_modal = torch.sum(attn, dim=1)
                # normalize
                attn = attn / norm_per_modal.unsqueeze_(1)
            else:
                raise ValueError("Not a valid attention calculation")
        return attn

    @staticmethod
    def replace_zero_rows_with_vector(tensor, p):
        mask = (tensor.sum(dim=1) == 0).unsqueeze(1)
        num_zero_rows = mask.sum()
        vector_size = tensor.size(1)
        vector = \
            torch.bernoulli(
                torch.full((num_zero_rows, vector_size), p, device="cuda")
            )
        replaced_tensor = tensor.masked_scatter(mask, vector)
        return replaced_tensor

    @staticmethod
    def replace_zero_one_rows_with_vector(tensor, p):
        mask_one = (tensor.sum(dim=1) == 1).unsqueeze(1)
        mask_zero = (tensor.sum(dim=1) == 0).unsqueeze(1)
        mask = mask_one + mask_zero # logical AND
        num_zero_rows = mask.sum()
        vector_size = tensor.size(1)
        vector = \
            torch.bernoulli(
                torch.full((num_zero_rows, vector_size), p, device="cuda")
            )
        replaced_tensor = tensor.masked_scatter(mask, vector)
        return replaced_tensor

    def get_mix_factors(self, bsz, n_mods, device):
        # sample alpha's for Dirichlet
        alphas = self.alpha_pdf.sample((bsz, ))
        # alphas = alphas.repeat(1, bsz)
        # sample p_mask (p_keep) for Dirichlet
        p_keep = self.mask_pdf.sample((self.n_samples, bsz))
        # (new_bsz, bsz)
        mixing_factors_pdf = dirichlet.Dirichlet(alphas)
        mask_pdf = bernoulli.Bernoulli(p_keep)
        # mixing_factors_pdf = \
        #     dirichlet.Dirichlet(alphas).sample((self.n_samples, ))

        # print("spot - 2 - untargeted")
        if self.iid_mixes:
            # print("spot - 3 - aniso")
            # sample a single mask for all modalitites
            mask = mask_pdf.sample()
            mask = self.replace_zero_one_rows_with_vector(
                mask, 0.5*(self.mask_low+self.mask_high)
            )
            # import pdb; pdb.set_trace()
            # TODO: better implementation when masking alphas
            lam = []
            for _ in range(n_mods):
                # mask = mask_pdf.sample()
                # mask = self.replace_zero_one_rows_with_vector(
                #     mask, 0.5*(self.mask_low+self.mask_high)
                # )
                # tmp_factors = mixing_factors_pdf.sample((self.n_samples, ))
                tmp_factors = mixing_factors_pdf.sample((self.n_samples, )) * mask
                norm_factor = torch.sum(tmp_factors, dim=1).unsqueeze_(1) + 1e-5 # smoothing
                lam.append(tmp_factors / norm_factor)
        else:
            # print("spot - 4 - iso")
            mask = mask_pdf.sample()
            mask = self.replace_zero_one_rows_with_vector(
                mask, 0.5*(self.mask_low+self.mask_high)
            )
            tmp_factors = \
                mixing_factors_pdf.sample((self.n_samples, )) * mask
            norm_factor = torch.sum(tmp_factors, dim=1).unsqueeze_(1) + 1e-5
            mixing_factors = tmp_factors / norm_factor
            lam = [mixing_factors for _ in range(n_mods)]

        # List[(N,B)]
        return lam

    @staticmethod
    def norm_mix_factors(mixing_factors, attn, n_mods):
        for k in range(n_mods):
            # reweight lambdas
            mixing_factors[k] = \
                mixing_factors[k] * attn[..., k].unsqueeze(0)
            # row normalization (N, B), summation over the B
            norm_factor = torch.sum(mixing_factors[k], dim=1) + 1e-5
            mixing_factors[k] = \
                mixing_factors[k] / norm_factor.unsqueeze_(1)
        return mixing_factors

    def convex_sums(self, mixing_factors, x, target, n_mods):
        """convex sum calculator function"""
        x_new = []
        y_new = []

        for k in range(n_mods):
            # (N,B) @ (B,D) --> (N, D)
            x_new.append(torch.matmul(mixing_factors[k], x[k]))
            # (N,B) @ (B,1) --> (N, 1)
            y_new.append(torch.matmul(
                mixing_factors[k], target.unsqueeze(1))
            )
        # aggregated label
        y_new = torch.sum(torch.stack(y_new, dim=1), dim=1) * (1/n_mods)
        y_new.squeeze_(1)  # (B,1) --> (B,)
        return x_new, y_new

    def mlabel_convex_sums(self, mixing_factors, x, target, n_mods):
        """multilabel convex sum calculator function"""
        x_new = []
        y_unimodal_new = []
        y_multimodal_new = []

        for k in range(n_mods):
            # (N,B) @ (B,D) --> (N, D)
            x_new.append(torch.matmul(mixing_factors[k], x[k]))
            # (N,B) @ (B,1) --> (N, 1)
            y_unimodal_new.append(torch.matmul(mixing_factors[k], target[k].unsqueeze(1)).squeeze_(1))
            y_multimodal_new.append(torch.matmul(mixing_factors[k], target[-1].unsqueeze(1)))
        # aggregated label
        y_multimodal_new = torch.sum(torch.stack(y_multimodal_new, dim=1), dim=1) * (1/n_mods)
        y_multimodal_new.squeeze_(1)  # (B,1) --> (B,)
        # collect both unimodal and multimodal to return
        y_unimodal_new.append(y_multimodal_new)
        return x_new, y_unimodal_new

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # print(f"Unilabel forward pass")
            # each tensor is [B, D], get necessary dimensions
            x_m = x[0]
            n_mods = len(x)
            bsz, d = x_m.size(0), x_m.size(1)

            with torch.no_grad():
                # calculate attentions, attn (B, 3)
                attn = self.get_attn(x, attn, bsz, d, n_mods)
                # print(f"Attnetions are {attn}")

                # calculate mixing factors, [(N, B), (N, B), (N, B)]
                mixing_factors = self.get_mix_factors(bsz, n_mods, x_m.device)

                # normalization step
                mixing_factors = self.norm_mix_factors(
                    mixing_factors, attn, n_mods
                )
                # print(f"Mixing factors are {mixing_factors}")

            # calculate convex sums
            x, target = self.convex_sums(mixing_factors, x, target, n_mods)

        return x, target

    def multilabel_forward(self, x, target, attn=None):
        """
        Multilabel version which assumes x: [text, audio, video]
        and y: [text, audio, video, multimodal]
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # print(f"Multilabel forward pass")
            # each tensor is [B, D], get necessary dimensions
            x_m = x[0]
            n_mods = len(x)
            bsz = x_m.size(0)

            with torch.no_grad():
                # calculate attentions, attn (B, 3)
                attn = self.mlabel_get_attn(x, attn, bsz, n_mods)

                # calculate mixing factors, [(N, B), (N, B), (N, B)]
                mixing_factors = self.get_mix_factors(bsz, n_mods, x_m.device)

                # normalization step
                mixing_factors = self.norm_mix_factors(
                    mixing_factors, attn, n_mods
                )
                # print(f"Mixing factors are {mixing_factors}")

            # calculate convex sums
            x, target = self.mlabel_convex_sums(mixing_factors, x, target, n_mods)

        return x, target


class CCM3xUp(nn.Module):
    def __init__(
            self,
            ci_a_target: float = 3.0,
            ci_a_other: float = 0.1,
            ci_target_m: int = 0,
            ci_n_samples: Optional[Union[int, None]] = None,
            ci_add_noise: Optional[bool] = False,
            ci_std: Optional[float] = 0.01,
            ci_reweight: Optional[str] = "uniform",
            ci_keep_target_label: Optional[bool] = False,
            ci_iid_mixes: Optional[bool] = False,
            ic_a_target: float = 3.0,
            ic_a_other: float = 0.1,
            ic_target_m: int = 0,
            ic_n_samples: Optional[Union[int, None]] = None,
            ic_add_noise: Optional[bool] = False,
            ic_std: Optional[float] = 0.01,
            ic_reweight: Optional[str] = "uniform",
    ):
        """Cross-Sample Cross-Modal Multimodal MixUp extension. (CC-M3xUp)
        This is the most general case and may be implemented in various ways.
        In this implementation we will combine CI-IC M3xUp by applying CI first
        and then IC. We do so because IC current implementation does not affect
        the label and therefore only CI affects it.
        Takes a batch and randomly permutes it. Then it mixes the two batches
        in a cross sample intra modal manner. After that an intra sample cross
        modal mixing is being performed.

        Args:
            ci_* : we refer to the CIM3xUp implementation
            ic_* : we refer to the ICM3xUp implementation
        """
        super(CCM3xUp, self).__init__()
        self.ci_m3xup = CIM3xUp(
            a_target=ci_a_target,
            a_other=ci_a_other,
            target_m=ci_target_m,
            n_samples=ci_n_samples,
            add_noise=ci_add_noise,
            std=ci_std,
            reweight=ci_reweight,
            keep_target_label=ci_keep_target_label,
            iid_mixes=ci_iid_mixes,
        )
        self.ic_m3xup = ICM3xUp(
            a_target=ic_a_target,
            a_other=ic_a_other,
            target_m=ic_target_m,
            n_samples=ic_n_samples,
            add_noise=ic_add_noise,
            std=ic_std,
            reweight=ic_reweight
        )

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression. The attn only effects CI module

        Args:
            x List[torch.tensor]: [m[B, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, 3)
        """
        if self.training:
            # print("Original tensor before m3xup was")
            # print(f"{x[0]}")
            # print(f"{x[1]}")
            # print("Labels before m3xup were")
            # print(target)
            x, target = self.ci_m3xup(x, target, attn)
            # print("Text after ci_m3xup is")
            # print(x[0])
            # print("Audio after ci_m3xup is")
            # print(x[1])
            # print("Labels after ci_m3xup are")
            # print(target)
            x, target = self.ic_m3xup(x, target)
            # print("Text after ic_m3xup is")
            # print(x[0])
            # print("Audio after ic_m3xup is")
            # print(x[1])
            # print("Labels after ic_m3xup are")
            # print(target)
        return x, target


class SeqICM3xUp(nn.Module):
    def __init__(
            self,
            a_target: float = 3.0,
            a_other: float = 0.1,
            target_m: int = 0,
            n_samples: Optional[Union[int, None]] = None,
            add_noise: Optional[bool] = False,
            std: Optional[float] = 0.01,
            reweight: Optional[str] = "uniform",
    ):
        """Intra-Sample Cross-Modal Multimodal MixUp Sequential extension.
        (IC-M3xUp-Seq) Assumes a target modality and mixes all three modalities
        to replace this target modality. Mixing occurs across all timesteps.
        The intra flag handles the case of intra sample mixing or cross sample
        mixing. The mixing occurs in a targeted manner. Assumes 3 modalities.

        Args:
            a_target (float): the first alpha (l2r) in the beta distribution
            a_other (float): the 2nd/3rd alpha (l2r) in the beta distribution
            target_m (int): 0->L, 1->A, 2->V
            n_samples (int): more than one mixes in a particular modality
            add_noise (bool): by default does not use the noisy mixup trick
            std (float): the std for the noisy mixup
            reweight (str): add label reweighting based on each vector
                contribution
        """
        super(SeqICM3xUp, self).__init__()
        self.a_target = a_target
        self.a_other = a_other
        self.target_m = target_m
        self.add_noise = add_noise
        self.std = std
        self.reweight = reweight

        alphas = torch.tensor([
            self.a_target,
            self.a_other,
            self.a_other
        ])
        # Assumes L,A,V order
        self.alphas = torch.roll(alphas, self.target_m)

        self.n_samples = n_samples if n_samples is not None else None

        # define the distribution from which we will sample lambda's
        if self.add_noise:
            self.additive_noise = normal.Normal(
                torch.tensor(0.0),
                torch.tensor(self.std)
            )

    @staticmethod
    def perm(x, target):
        return randperm(x, target)

    def get_mix_bsz(self, bsz):
        if self.n_samples is None:
            return bsz
        else:
            return self.n_samples

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x List[torch.tensor]: [m[B, L, D]] m tensors of (B,D) size
            target (torch.tensor): [B, ]
            attn Optional[Union[None, torch.tensor]]: (B, L, 3)
        """
        if self.training:
            # each tensor is [B, L, D]
            m_tar = x[self.target_m]
            n_mods = len(x)
            # l, a, v = x[0], x[1], x[2]
            bsz, maxlen, d = m_tar.size(0), m_tar.size(1), m_tar.size(2)
            # get attentions
            if self.reweight == "uniform":
                # normalized attention weights per timestep per modality
                attn = \
                    torch.ones((bsz, maxlen, n_mods)).to(m_tar.device) / maxlen
            elif self.reweight == "gap":
                attn = []
                for k in range(n_mods):
                    # (B,L,D)--> attn: (B,L)
                    attn.append(torch.sum(x[k], dim=2) / d)
                    # remove zero entries
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                    # norm per timestep (B,L) --> (B,)
                    norm_val = torch.sum(attn[k], dim=1)
                    attn[k] = attn[k] / norm_val.unsqueeze_(1)
                # attn: (B,L,3)
                attn = torch.stack(attn, dim=2)
            else:
                # already given attention
                # (B,L,3) and assumes normalized values per timestep
                if attn is None:
                    raise ValueError("give an attention argument at forward")
            # get new_bsz
            new_bsz = self.get_mix_bsz(bsz)
            # print(f"New batch size is {new_bsz} whereas old is {bsz}")
            if new_bsz > bsz:
                for k in range(n_mods):
                    x[k] = x[k].repeat(new_bsz//bsz + 1, 1)[:new_bsz]
            # get alphas
            # add option where we can sample alphas fromo uniforms
            # (N, 3)
            # alphas = self.alphas.to(m_tar.device).repeat(new_bsz, 1)
            alphas = self.alphas.to(m_tar.device)
            # expand attentions (B, L, 3) --> (N, L, 3)
            # if new_bsz > bsz:
            #     attn = attn.repeat(new_bsz // bsz + 1, 1)[:new_bsz]
            # sample dirichlet, already in (N, 3) format
            mixing_factors = \
                dirichlet.Dirichlet(alphas).sample((new_bsz * maxlen, ))
            # convert to (N, L, 3)
            mixing_factors = mixing_factors.reshape(new_bsz, maxlen, 3)
            # combine and normalize mixing_factors
            mixing_factors = mixing_factors * attn
            # renormalize per row
            norm_factors = torch.sum(mixing_factors, dim=2)
            mixing_factors = mixing_factors / norm_factors.unsqueeze_(2)
            # print(f"Mixing factors are {torch.sum(mixing_factors, dim=0)/1000}")
            # print(mixing_factors.shape)
            for k in range(n_mods):
                if k == 0:
                    # (N,L)*(N,L,3)
                    x_new = x[k] * mixing_factors[..., k].unsqueeze(2)
                else:
                    x_new = x_new + x[k] * mixing_factors[..., k].unsqueeze(2)
            if self.add_noise:
                x_new = x_new + \
                    self.additive_noise.sample(new_bsz, d, n_mods).to(m_tar.device)
            # permute to get random places for a_target in the matrix
            if new_bsz > bsz:
                perm = torch.randperm(new_bsz).to(m_tar.device)
                x_new = x_new[perm]
            # print(f"x_new shape is {x_new.shape}")
            for k in range(n_mods):
                if k == self.target_m:
                    x[k] = x_new
                else:
                    if new_bsz > bsz:
                        x[k] = x[k][perm]
            # fix target accordingly (B, ) --> (B,1 )
            # Comment: since i have assumed common label in both
            if new_bsz > bsz:
                target.unsqueeze_(1)
                target = target.repeat(new_bsz//bsz+1, 1)[:new_bsz]
                target = target[perm].reshape(-1)

        return x, target


class SeqCIM3xUp(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_low: float = 0.5,
            uni_high: Optional[float] = 2.0,
            reweight: Optional[str] = "uniform",
            targeted: Optional[bool] = False,
            target_m: Optional[Union[None, int]] = 0,
            a_target: Optional[float] = 3.0,
            a_other: Optional[float] = 0.1,
            iid_mixes: Optional[bool]= False,
            clf_all: Optional[bool] = False,
    ):
        """Sequential Cross Sample Intra Modal MixUp

        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
            reweight Optional(str): "uniform", "gap", "attn"
            targeted (bool): by default does not use the targeted approach
                where a target vector is held and the others are interpolated
                via small factors
            a_target (float): the concentration of the target sample, 3,4,5
            a_other (float): the concentration of the other sample, 0.1, 0.3
            clf_all (bool): when True return targets for all timesteps/tokens
                in the sequence. When False returns only one (MSA task).
        """
        super(SeqCIM3xUp, self).__init__()
        self.n_samples = n_samples
        self.uni_low = uni_low
        self.uni_high = uni_high
        self.target_m = target_m
        self.a_target = a_target
        self.a_other = a_other
        self.targeted = targeted
        self.reweight = reweight
        self.iid_mixes = iid_mixes
        self.clf_all = clf_all  # not implemented here

        if self.targeted:
            self.a_target = a_target
            self.a_other = a_other
        else:
            self.alpha_pdf = uniform.Uniform(
                low=torch.tensor(self.uni_low),
                high=torch.tensor(self.uni_high),
            )

    def get_mix_bsz(self, bsz):
        if self.n_samples is None or self.n_samples <= bsz:
            return bsz
        else:
            return self.n_samples

    def get_attn(self, attn, x):
        """Return the attention (B,L,3) tensor"""
        n_mods = len(x)
        bsz, maxlen, d = x[0].size(0), x[0].size(1), x[0].size(2)
        if attn is not None:
            return attn
        else:
            if self.reweight == "uniform":
                # normalized attention weights per timestep per modality
                attn = \
                    torch.ones((bsz, maxlen, n_mods)).to(x[0].device) / maxlen
            elif self.reweight == "gap":
                attn = []
                for k in range(n_mods):
                    # (B,L,D)--> attn: (B,L)
                    attn.append(torch.sum(x[k], dim=2) / d)
                    # remove zero entries
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                    # norm per timestep (B,L) --> (B,)
                    norm_val = torch.sum(attn[k], dim=1)
                    attn[k] = attn[k] / norm_val.unsqueeze_(1)
                # attn: (B,L,3)
                attn = torch.stack(attn, dim=2)
            else:
                # already given attention
                # (B,L,3) and assumes normalized values per timestep
                if attn is None:
                    raise ValueError("give an attention argument at forward")
            return attn

    def get_mix_factors(self, bsz, maxlen, n_mods, device):
        if self.targeted:
            # alphas are a pair --> (B,2)
            # we assume fixed alphas for targeted
            alphas = \
                torch.tensor(
                    [self.a_target, self.a_other]
                ).to(device).repeat(bsz, 1)
            mixing_factors_pdf = dirichlet.Dirichlet(alphas)

            # handle single target modality case
            if self.target_m is not None:
                one_hot_mix = torch.tensor([1, 0]).to(device)
                # [1, 0] in shape (1, 1, 2)
                one_hot_mix.unsqueeze_(1).unsqueeze_(2)
                one_hot_mix = one_hot_mix.permute(1, 2, 0)
                # generate mixing_factors for each modality
                lam = [one_hot_mix for _ in range(n_mods)]
                lam[self.target_m] = \
                    mixing_factors_pdf.sample((maxlen, )).permute(1, 0, 2)
            else:  # no target modality scenario
                if self.iid_mixes:
                    lam = \
                        [
                            mixing_factors_pdf.sample(
                                (maxlen,)
                            ).reshape(bsz, maxlen, 2) for _ in range(n_mods)
                        ]
                else:
                    mix_factors = mixing_factors_pdf.sample(
                        (maxlen, )
                    ).permute(1, 0, 2)
                    lam = [mix_factors for _ in range(n_mods)]
        else:
            # sample alpha's from uniform in the untargeted case
            # (bsz, )
            alphas = self.alpha_pdf.sample((bsz, 2)).to(device)
            # (B, 2)
            mixing_factors_pdf = dirichlet.Dirichlet(alphas)

            # handle target modality scenario
            if self.target_m is not None:
                # sample a different lam for every timestep of the target seq
                # (B, L, 2)
                mixing_factors = \
                    mixing_factors_pdf.sample((maxlen, )).permute(1, 0, 2)
                one_hot_mix = torch.tensor([1, 0]).to(device)
                one_hot_mix.unsqueeze_(1).unsqueeze_(2)
                # (1, 1, 2) to avoid extra memory
                one_hot_mix = one_hot_mix.permute(1, 2, 0)
                lam = [one_hot_mix for _ in range(n_mods)]
                lam[self.target_m] = mixing_factors
            else:
                print(f"This scenario should not be considered since it mixes too much")
                print(f"Maybe the case where a single lam is being sampled across all timesteps is better")
                # uncomment to get single lambda for the whole sequence
                # if self.iid_mixes:
                #     lam = [
                #         mixing_factors_pdf.sample()  # (B,2)
                #         for _ in range(n_mods)
                #     ]

                if self.iid_mixes:
                    lam = [
                        mixing_factors_pdf.sample((maxlen, )).permute(1, 0, 2)
                        for _ in range(n_mods)
                    ]
                else:
                    # similar as above for a single lam per seq do uncomment
                    # mixing_factors = mixing_factors_pdf.sample()  # (B, 2)
                    # (B, L, 2)
                    mixing_factors = \
                        mixing_factors_pdf.sample((maxlen, )).permute(1, 0, 2)
                    lam = [mixing_factors for _ in range(n_mods)]

            # List[ (B, L, 2) || (1, 1, 2) ]
        return lam

    @staticmethod
    def expand_mm(x, target, n_samples):
        x_new = []
        bsz = x[0].size(0)
        for x_mm in x:
            x_new.append(
                x_mm.repeat(n_samples//bsz + 1, 1, 1)[:n_samples]
            )
        target = target.repeat(n_samples//bsz + 1)[:n_samples]
        return x_new, target

    @staticmethod
    def expand_attn(attn, n_samples):
        bsz = attn.size(0)
        # (B,L,3) --> (N,L,3)
        attn = attn.repeat(n_samples//bsz + 1, 1, 1)[:n_samples]
        return attn

    @staticmethod
    def expand_mix_factors(mix_factors, n_samples, bsz):
        new_mix_f = []
        for mix_f in mix_factors:
            if mix_f.size(0) == bsz:
                new_mix_f.append(
                    mix_f.repeat(n_samples//bsz + 1, 1, 1)[:n_samples]  # (N,L,2)
                )
            else:
                new_mix_f.append(mix_f)
        return new_mix_f

    @staticmethod
    def get_perm(bsz, device):
        return torch.randperm(bsz).to(device)

    @staticmethod
    def norm_mix_factors(mixing_factors, attn, perm):
        """Gets (N, L, 2) mixing factors and (N, L) per modality
        addiitonal (N) perm to generate second view of the batch
        """
        # attn: (N,L,3)
        attn_perm = attn[perm]
        norm_mf = []
        for k, mf in enumerate(mixing_factors):
            # mf: (N, L, 2)
            # m_attn: (N, L ,2)
            m_attn = torch.stack(
                [attn[..., k], attn_perm[..., k]], dim=2
            )
            mf = mf * m_attn
            # norm across the mixing dimension --> 2
            norm_factor = torch.sum(mf, dim=2)
            mf = mf / norm_factor.unsqueeze_(2)
            norm_mf.append(mf)
        return norm_mf

    def mix_step(self, x, y, mixing_factors, perm, new_bsz, bsz):
        new_x = []
        new_y = []
        # y: (B, 1)
        y = y.unsqueeze(1)
        y_perm = y[perm]
        y_combined = torch.stack([y, y_perm], dim=2)
        for k, x_mm in enumerate(x):
            #  mf: (N, L, 2)
            mf = mixing_factors[k]
            x_mm_perm = x_mm[perm]
            # x_mm_plus: (N, L, 2)
            x_mm_combined = torch.stack([x_mm, x_mm_perm], dim=3)
            # elementwise multiplication and sum (commutative): (N, L)
            new_x.append(
                torch.einsum(
                    '...ij,...kj->...i',
                    x_mm_combined,
                    mf.unsqueeze(2)
                )
            )
            # average over L: timestep dimension
            new_y.append(
                torch.mean(torch.einsum('blc, bdc->bl', mf, y_combined), dim=1)
            )
        # mean label over three modalities
        y_agg = torch.mean(torch.stack(new_y, dim=1), dim=1)

        # re-perm to avoid label repetition in some cases
        if new_bsz > bsz:
            for k in range(len(x)):
                new_x[k] = new_x[k][perm]
            y_agg = y_agg[perm]

        return new_x, y_agg

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x: (torch.tensor): List[[B, L, D]]
            target: (torch.tensor): [B, ]
            attn: Optional[torch.tensor]: [B, L] the attention weights to
                reweight the tokens in the sequence
        """
        if self.training:
            # eaxh tensor [B, L, D]
            # get required dimensions
            bsz, maxlen, d = x[0].size(0), x[0].size(1), x[0].size(2)
            n_mods = len(x)

            # get new_bsz
            new_bsz = self.get_mix_bsz(bsz)
            print(f"The new bsz is {new_bsz}")

            # get attentions: (B, L, 3)
            attn = self.get_attn(attn, x)
            print(f"The attention is {attn}")

            # get mixing_factors: 3-len List[(B, L, 2) || (1, 1, 2)]
            mixing_factors = self.get_mix_factors(
                bsz, maxlen, n_mods, x[0].device
            )
            print(f"The mixing factors are {mixing_factors}")

            # expansion step: from B to N
            if new_bsz > bsz:
                x, target = self.expand_mm(x, target, new_bsz)
                mixing_factors = self.expand_mix_factors(
                    mixing_factors, new_bsz, bsz
                )
                attn = self.expand_attn(attn, new_bsz)

            # get a permutation
            perm = self.get_perm(new_bsz, x[0].device)

            # normalization step
            mixing_factors = self.norm_mix_factors(mixing_factors, attn, perm)
            print(f"The normalized mixing factors are {mixing_factors}")

            # mixing step
            x, target = \
                self.mix_step(x, target, mixing_factors, perm, new_bsz, bsz)

            return x, target


class SeqConvexCIM3xUp(nn.Module):
    def __init__(
            self,
            n_samples: int = 1000,
            uni_low: float = 0.5,
            uni_high: Optional[float] = 2.0,
            reweight: Optional[str] = "uniform",
            targeted: Optional[bool] = False,
            target_m: Optional[Union[None, int]] = 0,
            a_target: Optional[float] = 3.0,
            a_other: Optional[float] = 0.1,
            iid_mixes: Optional[bool] = False,
            clf_all: Optional[bool] = False,
    ):
        """Sequential Convex Cross Sample Intra Modal MixUp

        Args:
            n_samples (int): the number of convex interpolations
            uni_low (float): the low bound of the uniform distribution
            uni_high (float): the high bound of the uniform distribution
            reweight Optional(str): "uniform", "gap", "attn"
            targeted (bool): by default does not use the targeted approach
                where a target vector is held and the others are interpolated
                via small factors
            a_target (float): the concentration of the target sample, 3,4,5
            a_other (float): the concentration of the other sample, 0.1, 0.3
            clf_all (bool): when True return targets for all timesteps/tokens
                in the sequence. When False returns only one (MSA task).
        """
        super(SeqConvexCIM3xUp, self).__init__()
        self.n_samples = n_samples
        self.uni_low = uni_low
        self.uni_high = uni_high
        self.target_m = target_m
        self.a_target = a_target
        self.a_other = a_other
        self.targeted = targeted
        self.reweight = reweight
        self.iid_mixes = iid_mixes
        self.clf_all = clf_all  # not implemented here

        if self.targeted:
            self.a_target = a_target
            self.a_other = a_other
        else:
            self.alpha_pdf = uniform.Uniform(
                low=torch.tensor(self.uni_low),
                high=torch.tensor(self.uni_high),
            )

    def get_mix_bsz(self, bsz):
        if self.n_samples is None or self.n_samples <= bsz:
            return bsz
        else:
            return self.n_samples

    def get_attn(self, attn, x):
        """Return the attention (B,L,3) tensor"""
        n_mods = len(x)
        bsz, maxlen, d = x[0].size(0), x[0].size(1), x[0].size(2)
        if attn is not None:
            return attn
        else:
            if self.reweight == "uniform":
                # normalized attention weights per timestep per modality
                attn = \
                    torch.ones((bsz, maxlen, n_mods)).to(x[0].device) / maxlen
            elif self.reweight == "gap":
                attn = []
                for k in range(n_mods):
                    # (B,L,D)--> attn: (B,L)
                    attn.append(torch.sum(x[k], dim=2) / d)
                    # remove zero entries
                    attn[k] = attn[k].clamp(min=0.0) + 1e-5
                    # norm per timestep (B,L) --> (B,)
                    norm_val = torch.sum(attn[k], dim=1)
                    attn[k] = attn[k] / norm_val.unsqueeze_(1)
                # attn: (B,L,3)
                attn = torch.stack(attn, dim=2)
            else:
                # already given attention
                # (B,L,3) and assumes normalized values per timestep
                if attn is None:
                    raise ValueError("give an attention argument at forward")
            return attn

    def get_mix_factors(self, new_bsz, bsz, maxlen, n_mods, device):
        if self.targeted:
            # create diagonal
            alphas = (torch.eye(bsz).to(device)) \
                * (self.a_target - self.a_other)
            # expand alphas
            alphas = alphas.repeat(new_bsz//bsz + 1, 1)[:new_bsz]
            # add self.a_other everywhere
            alphas += self.a_other
            # permute to get random places for a_target in the matrix
            perm = torch.randperm(new_bsz).to(device)
            alphas = alphas[perm]
            # when sampling in (N, B) format
            mixing_factors_pdf = dirichlet.Dirichlet(alphas)

            # handle single target modality case
            if self.target_m is not None:
                one_hot_mix = torch.eye(bsz).to(device)
                one_hot_mix = \
                    one_hot_mix.repeat(new_bsz//bsz+1, 1)[:new_bsz]
                # permuted (N, B, 1)
                one_hot_mix = one_hot_mix[perm].unsqueeze(2)

                # generate mixing_factors for each modality
                lam = [one_hot_mix for _ in range(n_mods)]
                # (N, B, L)
                lam[self.target_m] = \
                    mixing_factors_pdf.sample((maxlen, )).permute(1, 2, 0)
            else:  # no target modality scenario
                if self.iid_mixes:
                    # (N, B, L)
                    lam = \
                        [
                            mixing_factors_pdf.sample(
                                (maxlen,)
                            ).permute(1, 2, 0) for _ in range(n_mods)
                        ]
                else:
                    # (N, B, L)
                    mix_factors = mixing_factors_pdf.sample(
                        (maxlen, )
                    ).permute(1, 2, 0)
                    lam = [mix_factors for _ in range(n_mods)]
        else:
            # sample alpha's from uniform in the untargeted case
            # (B, )
            alphas = self.alpha_pdf.sample((bsz, )).to(device)
            # (N, B)
            mixing_factors_pdf = dirichlet.Dirichlet(alphas)

            # handle target modality scenario
            if self.target_m is not None:
                # sample a different lam for every timestep of the target seq
                # (N, B, L)
                mixing_factors = \
                    mixing_factors_pdf.sample((new_bsz, maxlen)).permute(0, 2, 1)
                ## Case-1 --> find max per timestep and get this mixed sequence
                # max_indices = torch.argmax(mixing_factors, dim=1)
                # one_hot_mix = \
                #     torch.eye(mixing_factors.shape[1], device=device)[max_indices]
                # Case-2 --> find max of aggregated timestep --> one-hot analogy
                # I implement case-2 here
                sum_mf = torch.sum(mixing_factors, dim=2) / maxlen  # (N, B)
                max_indices = torch.argmax(sum_mf, dim=1)
                one_hot_mix = \
                    torch.eye(sum_mf.shape[1], device=device)[max_indices]
                # (N, B, 1)
                one_hot_mix.unsqueeze_(2)
                lam = [one_hot_mix for _ in range(n_mods)]
                lam[self.target_m] = mixing_factors
            else:
                print(f"This scenario should not be considered since it mixes too much")
                print(f"Maybe the case where a single lam is being sampled across all timesteps is better")
                ######### uncomment to get single lambda for the whole sequence
                # if self.iid_mixes:
                #     lam = [
                #         mixing_factors_pdf.sample()  # (B,2)
                #         for _ in range(n_mods)
                #     ]

                if self.iid_mixes:
                    lam = [
                        mixing_factors_pdf.sample((new_bsz, maxlen)).permute(0, 2, 1)
                        for _ in range(n_mods)
                    ]
                else:
                    #### similar as above for a single lam per seq do uncomment
                    # mixing_factors = mixing_factors_pdf.sample()  # (B, 2)
                    # (N, B, L)
                    mixing_factors = \
                        mixing_factors_pdf.sample((new_bsz, maxlen)).permute(0, 2, 1)
                    lam = [mixing_factors for _ in range(n_mods)]

            # List[(N, B, L) || (N, B, 1)]
        return lam

    @staticmethod
    def expand_mm(x, target, n_samples):
        x_new = []
        bsz = x[0].size(0)
        for x_mm in x:
            x_new.append(
                x_mm.repeat(n_samples//bsz + 1, 1, 1)[:n_samples]
            )
        target = target.repeat(n_samples//bsz + 1)[:n_samples]
        return x_new, target

    @staticmethod
    def expand_attn(attn, n_samples):
        bsz = attn.size(0)
        # (B,L,3) --> (N,L,3)
        attn = attn.repeat(n_samples//bsz + 1, 1, 1)[:n_samples]
        return attn

    @staticmethod
    def expand_mix_factors(mix_factors, n_samples, bsz):
        new_mix_f = []
        for mix_f in mix_factors:
            if mix_f.size(0) == bsz:
                new_mix_f.append(
                    mix_f.repeat(n_samples//bsz + 1, 1, 1)[:n_samples]  # (N,L,2)
                )
            else:
                new_mix_f.append(mix_f)
        return new_mix_f

    @staticmethod
    def get_perm(bsz, device):
        return torch.randperm(bsz).to(device)

    @staticmethod
    def norm_mix_factors(mixing_factors, attn):
        """Get List[(N, B, L),...] of 3 mixing factors
        and (B, L, 3) attn per modality
        """
        # mixing_factors: [(N, B, L) || (N, B, 1)]
        # attn: (B, L, 3)
        norm_mf = []
        for k, mf in enumerate(mixing_factors):
            # mf: (N, B, L) || (N, B, 1)
            # attn[..., k]: (1, B, L)
            mf = mf * attn[..., k].unsqueeze(0)
            # norm across the mixing (batch) dimension --> 1, (N, L)
            norm_factor = torch.sum(mf, dim=1)
            mf = mf / norm_factor.unsqueeze_(1)
            norm_mf.append(mf)
        return norm_mf

    def mix_step(self, x, y, mixing_factors, maxlen):
        new_x = []
        new_y = []
        # y: (B, 1)
        y = y.unsqueeze(1)
        for k, x_mm in enumerate(x):
            #  x_mm: (B, L, D)
            #  mf: (N, B, L)
            mf = mixing_factors[k]
            # tmp_x = x_mm.permute(1, 2, 0)
            tmp_x = x_mm
            # elementwise multiplication and sum (commutative): (N, L)
            new_x.append(
                torch.einsum('bld,nbl->nld', tmp_x, mf)
            )
            # average over L: timestep dimension
            new_y.append(
                torch.einsum('nbl,bl->n', mf, y) / maxlen
            )
        # mean label over three modalities
        y_agg = torch.mean(torch.stack(new_y, dim=1), dim=1)

        # # re-perm to avoid label repetition in some cases
        # if new_bsz > bsz:
        #     for k in range(len(x)):
        #         new_x[k] = new_x[k][perm]
        #     y_agg = y_agg[perm]

        return new_x, y_agg

    def forward(self, x, target, attn=None):
        """
        Assumes the targets are already one-hot for classification and
        continuous for regression.

        Args:
            x: (torch.tensor): List[[B, L, D]]
            target: (torch.tensor): [B, ]
            attn: Optional[torch.tensor]: [B, L] the attention weights to
                reweight the tokens in the sequence
        """
        if self.training:
            # eaxh tensor [B, L, D]
            # get required dimensions
            bsz, maxlen, d = x[0].size(0), x[0].size(1), x[0].size(2)
            n_mods = len(x)

            # get new_bsz
            new_bsz = self.get_mix_bsz(bsz)
            print(f"The new bsz is {new_bsz}")

            # get attentions: (B, L, 3)
            attn = self.get_attn(attn, x)
            print(f"The attention is {attn}")

            # get mixing_factors: 3-len List[(Ν, Β, L) || (N, B, 1)]
            mixing_factors = self.get_mix_factors(
                new_bsz, bsz, maxlen, n_mods, x[0].device
            )
            print(f"The mixing factors are {mixing_factors}")
            print(f"The text mixing factors shape is {mixing_factors[0].shape}")
            print(f"The audio mixing factors shape is {mixing_factors[1].shape}")

            # # expansion step: from B to N
            # if new_bsz > bsz:
            #     x, target = self.expand_mm(x, target, new_bsz)
            #     mixing_factors = self.expand_mix_factors(
            #         mixing_factors, new_bsz, bsz
            #     )
            #     attn = self.expand_attn(attn, new_bsz)

            # # get a permutation
            # perm = self.get_perm(new_bsz, x[0].device)

            # normalization step
            mixing_factors = self.norm_mix_factors(mixing_factors, attn)
            print(f"The normalized mixing factors are {mixing_factors}")

            # mixing step
            x, target = \
                self.mix_step(x, target, mixing_factors, maxlen)

            return x, target




if __name__ == "__main__":
    device = torch.device("cuda")
    # device = torch.device("cpu")

    # B, D = 4, 8
    # x = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device)
    # y = torch.arange(B).to(device)

    # # test MixUp implementation
    # print(f"Testing the implementation of MixUp with noise addition")
    # mixup = MixUp(1.0, add_noise=True, std=0.01).to(device)
    # x_new, y_new = mixup(x, y)
    # print(x_new)

    # # test SeqMixUp implementation
    # print(f"Testing the implementation of SeqMixUp with noise addition")
    # B, L, D = 4, 6, 8
    # x = \
    #     (torch.arange(B).unsqueeze_(1).unsqueeze_(2) * torch.ones((B, L, D))).to(device)
    # y = torch.arange(B).to(device)
    # seqmixup = SeqMixUp(1.0, add_noise=True)
    # x_new, y_new = seqmixup(x, y)
    # print(x_new)

    # # test MixUp implementation
    # print(f"Testing the implementation of ConvexUp")
    # B, D = 4, 8
    # x = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device)
    # # x = (torch.arange(B).unsqueeze(1).float()).to(device)
    # y = torch.arange(B).to(device).float()
    # cmixup = ConvexMixUp(n_samples=16, targeted=False).to(device)
    # x_new, y_new = cmixup(x, y)
    # print(f"Original input was {x}")
    # print(f"Original target was {y}")
    # print(f"New input is {x_new}")
    # print(f"New target is {y_new}")

    # test IC vector M3Xup
    B, D = 4, 8
    l = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device)
    a = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device) * 5
    v = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device) * 6
    y = torch.arange(B).to(device).float()
    # a_target = [1.0, 2.0]
    # a_other = [0.1, (0.1, 0.2, 0.25, 0.3)]
    m3xup = ICM3xUp(
        a_target=2.0, a_other=0.2, target_m=0, n_samples=10, reweight="gap",
    ).to(device)
    x_new, y_new = m3xup([l,a,v], y)
    print("Original tensor before m3xup was")
    print(f"{l}")
    print(f"{a}")
    print("Labels before m3xup were")
    print(y)
    print("Text after m3xup is")
    print(x_new[0])
    print("Audio after m3xup is")
    print(x_new[1])
    print("Labels after m3xup are")
    print(y_new)

    # # test CI vector M3Xup
    # B, D = 4, 8
    # l = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device)
    # a = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device) * 5
    # v = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device) * 6
    # y = torch.arange(B).to(device).float()
    # # a_target = [1.0, 2.0]
    # # a_other = [0.1, (0.1, 0.2, 0.25, 0.3)]
    # m3xup = CIM3xUp(
    #     a_target=1.0, a_other=1.0, target_m=None, n_samples=2*B,
    #     reweight="uniform", keep_target_label=False, iid_mixes=True
    # ).to(device)
    # x_new, y_new = m3xup([l,a,v], y)
    # print("Original tensor before m3xup was")
    # print(f"{l}")
    # print(f"{a}")
    # print("Labels before m3xup were")
    # print(y)
    # print("Text after m3xup is")
    # print(x_new[0])
    # print("Audio after m3xup is")
    # print(x_new[1])
    # print("Labels after m3xup are")
    # print(y_new)


    # # test CC vector M3Xup
    # B, D = 4, 8
    # l = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device)
    # a = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device) * 5
    # v = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device) * 6
    # y = torch.arange(B).to(device).float()
    # # a_target = [1.0, 2.0]
    # # a_other = [0.1, (0.1, 0.2, 0.25, 0.3)]
    # m3xup = CCM3xUp(
    #     ci_a_target=2.0, ci_a_other=.2, ci_target_m=None, ci_n_samples=None,
    #     ci_reweight="gap", ci_keep_target_label=False, ci_iid_mixes=True,
    #     ic_a_target=2.0, ic_a_other=.1, ic_target_m=0, ic_n_samples=None,
    #     ic_reweight="uniform"
    # ).to(device)
    # x_new, y_new = m3xup([l,a,v], y)

    # # test ConvexCI vector M3Xup
    # B, D = 4, 6
    # l = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device)
    # a = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device) * 5
    # v = (torch.arange(B).unsqueeze(1) * torch.ones((B, D))).to(device) * 6
    # y = torch.arange(B).to(device).float()
    # # a_target = [1.0, 2.0]
    # # a_other = [0.1, (0.1, 0.2, 0.25, 0.3)]
    # print("Original tensor before m3xup was")
    # print(f"{l}")
    # print(f"{a}")
    # print("Labels before m3xup were")
    # print(y)
    # m3xup = ConvexCIM3xUp(
    #     n_samples=8, targeted=True, a_target=2.0, a_other=0.1, target_m=None,
    #     reweight="gap", iid_mixes=True
    # ).to(device)
    # x_new, y_new = m3xup([l,a,v], y)
    # print("Text after m3xup is")
    # print(x_new[0])
    # print("Audio after m3xup is")
    # print(x_new[1])
    # print("Labels after m3xup are")
    # print(y_new)
    # print(y_new.shape)

    # # test IC-Seq implementation
    # print(f"Testing the implementation of ICM3xUp with noise addition")
    # B, L, D = 4, 6, 8
    # x = \
    #     (torch.arange(B).unsqueeze_(1).unsqueeze_(2) * torch.ones((B, L, D))).to(device)
    # x = [x*(k+1) for k in range(3)]
    # y = torch.arange(B).to(device)
    # print(f"Text before is {x[0]}")
    # print(f"Audio before is {x[1]}")
    # print(f"Target before is {y}")
    # icseq = ICM3xUpSeq(
    #     a_target=2.0, a_other=0.1, target_m=0, n_samples=None, add_noise=False,
    #     reweight="gap"
    # )
    # x, y = icseq(x, y)
    # print(f"Text after is {x[0]}")
    # print(f"Audio after is {x[1]}")
    # print(f"Target after is {y}")

    # # test CI-Seq implementation
    # print(f"Testing the implementation of CIM3xUp")
    # B, L, D = 4, 6, 8
    # x = \
    #     (torch.arange(B).unsqueeze_(1).unsqueeze_(2) * torch.ones((B, L, D))).to(device)
    # x = [x*(k+1) for k in range(3)]
    # y = torch.arange(B).to(device).float()
    # print(f"Text before is {x[0]}")
    # print(f"Audio before is {x[1]}")
    # print(f"Target before is {y}")
    # ciseq = SeqCIM3xUp(
    #     n_samples=2*B, uni_low=0.5, uni_high=2.0, reweight="gap",
    #     targeted=False, a_target=2.0, a_other=0.1, target_m=0,
    #     iid_mixes=False
    # )
    # x, y = ciseq(x, y)
    # print(f"Text after is {x[0]}")
    # print(f"Audio after is {x[1]}")
    # print(f"Target after is {y}")

    # # test Convex-CI-Seq implementation
    # print(f"Testing the implementation of SeqConvex CIM3xUp")
    # B, L, D = 4, 6, 8
    # x = \
    #     (torch.arange(B).unsqueeze_(1).unsqueeze_(2) * torch.ones((B, L, D))).to(device)
    # x = [x*(k+1) for k in range(3)]
    # y = torch.arange(B).to(device).float()
    # print(f"Text before is {x[0]}")
    # print(f"Audio before is {x[1]}")
    # print(f"Target before is {y}")
    # ciseq = SeqConvexCIM3xUp(
    #     n_samples=2*B, uni_low=0.5, uni_high=2.0, reweight="gap",
    #     targeted=False, a_target=2.0, a_other=0.1, target_m=0,
    #     iid_mixes=True
    # )
    # x, y = ciseq(x, y)
    # print(f"Text after is {x[0]}")
    # print(f"Audio after is {x[1]}")
    # print(f"Target after is {y}")
    # # import pdb; pdb.set_trace()

