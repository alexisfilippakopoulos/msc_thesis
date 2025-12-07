import math
import random
from tkinter import NO
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
import torch.distributions as D
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from MMSA.models.singleTask.MMGPT import (AV_Enc,
                                          MultiheadCrossAttention,
                                          LoRA_MLP, LayerNorm, MLP)

###################################################################################################
# Helper Code
###################################################################################################

def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work

def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "smolLM": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}

def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


###################################################################################################
# Core Layers
###################################################################################################
class SoftPerm(nn.Module):
    def __init__(
        self,
        p_apply: float = 0.5,
        p_feat: float = 0.2,
        maxlen: Optional[int] = 50,
        n_bn: Optional[int] = 16,
    ):
        """
        SoftPerm_Fast: SoftPermutation (Fast), randomly samples a Bernoulli distribution
        with hyperparam $p$. By sampling, it gets the `copy_area`
        (float), which denotes the proportion (=area) of feature dimensions
        that are going to be resampled (i.e how many features are going to be
        permuted).

        Args:
            p_t_mod (List[float]): Resample probabilities for each modality across timesteps.
                default=1's which means that we resample at every timestep
            alpha (List[float]): used in sampling uniform distribution for each modality.
                Encodes the amount of features to be resampled. In order to resample the same
                distribution one should give a list of multiple variables as input in [0,1).
            maxlen (int): must be given as input and refers to the maximum sequence length
                In this implementation we also assume that the sequences are already zero-padded
        """
        super().__init__()
        self.p_apply = p_apply
        self.p_feat = p_feat  # resampled features
        self.maxlen = maxlen
        self.n_bn = n_bn
        
        # weights
        self.w = torch.tensor([1/maxlen]*maxlen, device="cuda")
        # probs --> '1', non-resampled features
        self.bern = D.bernoulli.Bernoulli(probs=1 - torch.tensor(self.p_feat, device="cuda"))

    def forward(self, x):
        """
        Fast SoftPerm forward implementation
        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations
            m_ra (int): repeated augmentation index, default = 1
            real_len (torch.Tensor): [B] tensor of ints with the unpadded seq len
        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        # List of [B, L, D] and T,A,V order is assumed
        if self.training and self.p_feat > 0 and self.p_apply > random.random():
            bsz, seqlen, d = x.size()
            # w_perm = self.w.repeat(bsz, 1)
            # import pdb; pdb.set_trace()
            permutation = torch.multinomial(self.w, seqlen - self.n_bn)
            x_lang = x[:, :(seqlen-self.n_bn), :]
            x_perm = x_lang[:, permutation, :]
            # import pdb; pdb.set_trace()
            area_mask = self.bern.sample((bsz, d))
            # (B,D) --> (B, 1, D)
            area_mask = area_mask.unsqueeze(1)
            x_lang = area_mask * x_lang + (1 - area_mask) * x_perm
            x[:, :(seqlen-self.n_bn), :] = x_lang

        return x
    

class _LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
# class LayerNorm(nn.Module):
#     """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

#     def __init__(self, ndim, bias):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(ndim))
#         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

#     def forward(self, input):
#         input = input.to(torch.float32)
#         # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class _LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.get("n_up", 4 * self.hidden_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # if self.config.pretraining_tp > 1:
        #     slice = self.intermediate_size // self.config.pretraining_tp
        #     gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        #     up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        #     down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        #     gate_proj = torch.cat(
        #         [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        #     )
        #     up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        #     intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        #     down_proj = [
        #         F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        #     ]
        #     down_proj = sum(down_proj)
        # else:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    
class LoRALinear(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Actual trainable parameters
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D_in) --> (B, L, D_out)
        # debugging
        x = self.lora_dropout(x)
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = x * self.scaling
        # result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        # return result
        return x


class _LlamaMLP_LoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.get("n_up", 4 * self.hidden_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        #Lora params
        self.lora_gate_proj = LoRALinear(
            self.hidden_size,
            self.intermediate_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )
        self.lora_up_proj = LoRALinear(
            self.hidden_size,
            self.intermediate_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )
        self.lora_down_proj = LoRALinear(
            self.intermediate_size,
            self.hidden_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )

    def forward(self, x):
        x_gate = self.act_fn(self.gate_proj(x) + self.lora_gate_proj(x))
        x_up = self.up_proj(x) + self.lora_up_proj(x)
        x_down = x_gate + x_up
        x_down = self.down_proj(x_down) + self.lora_down_proj(x_down)
        return x_down



class MMBlock(nn.Module):
    """extension of the Block class to multimodal gated cross attention
    """
    def __init__(self, config, idx=-1):
        super().__init__()
        self.idx = idx
        self.lm_flavor = config.get("type", "llama")
        if "gpt" in self.lm_flavor:
            self.kdim = config.get("kv_dim", config.n_embd)
            self.ln_1 = LayerNorm(config.n_embd, config.bias)
            self.ln_2 = LayerNorm(config.n_embd, config.bias)
            self.attn = MultiheadCrossAttention(
                config.n_head,
                config.n_embd,  # the same as query size
                self.kdim,  # multimodal dimension, k, v
                config.dropout,
            )
            if config.use_lora:
                self.mlp = LoRA_MLP(config)
            else:
                self.mlp = MLP(config)
        else: 
            self.ln_1 = _LlamaRMSNorm(config.n_embd)
            # self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.kdim = config.get("kv_dim", config.n_embd)
            self.attn = MultiheadCrossAttention(
                config.n_head,
                config.n_embd,  # the same as query size
                self.kdim,  # multimodal dimension, k, v
                config.dropout,
            )
            self.ln_2 = _LlamaRMSNorm(config.n_embd)
            # self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            # if config.get("use_lora", False):
            #     self.mlp = LoRA_MLP(config)
            # else:
            self.mlp = _LlamaMLP_LoRA(config) # adopt this to initialize at the same
        
        ###########################################################################################
        # common parameters
        print(f"Ongoing with ----- {config.gating} ----- gating")
        self.gate_1 = nn.Sigmoid()
        self.gate_2 = nn.Sigmoid()
        # self.gate_1 = nn.Tanh()
        # self.gate_2 = nn.Tanh()
        # # init gate at ~0.5
        init_value = config.get("init_gate", 0)
        print(f"idx is {idx}")
        print(f"{init_value}")
        if config.get("init_gate_2", None):
            init_value_2 = config.get("init_gate_2")
        else:
            init_value_2 = init_value
        if self.idx == -1:
            self.alpha_1 = nn.Parameter(init_value * torch.ones(1))
            # self.alpha_1 = nn.Parameter(torch.zeros(1))
            self.alpha_2 = nn.Parameter(init_value_2 * torch.ones(1))
            # self.alpha_2 = nn.Parameter(torch.zeros(1))
        else:
            self.alpha_1 = nn.Parameter(init_value[idx] * torch.ones(1))
            self.alpha_2 = nn.Parameter(init_value_2[idx] * torch.ones(1))

        # combined cross-attention
        self.combine = config.get("combine", False)
        
    def forward(self, x_q, x_kv, x_prev=None):
        """
        x_q: the text-modality queries [B, L, D]
        x_k, x_v: the encoder's keys and values (context)
        """
        # (B,L,D)
        # for use_cache = True
        # x_q, cache = x_q[0], x_q[1]
        x_q = x_q[0]
        norm_x_q = self.ln_1(x_q)
        
        # cross-modal cross-attention
        x_ca = self.attn(norm_x_q, x_kv)
        
        # PAPER: gated cross-attention version
        x = x_q + self.gate_1(self.alpha_1) * x_ca
        
        # FFW of the fused info
        x = x + self.gate_2(self.alpha_2) * self.mlp(self.ln_2(x))
        
        # for cached implementation
        x = (x,)
        return x


class msaLMLayer(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native decoder LMLayer.
    """

    def __init__(
        self, ca_layer, decoder_layer, max_len=50,
        lm_flavor="llama", ldrop=0.0,
    ):
        super().__init__()
        self.ca_layer = ca_layer
        self.decoder_layer = decoder_layer
        self.max_len = max_len
        self.av_x = None
        self.lm_flavor = lm_flavor
        # self.ldrop = ldrop
        self.tie_ffw()
    
    def tie_ffw(self):
        if self.ca_layer is not None:
            print("COpying---------------------------------")
            if "gpt" in self.lm_flavor:
                # print(self.decoder_layer)
                assert self.ca_layer.mlp.c_fc.weight.shape == self.decoder_layer.mlp.c_fc.weight.shape[::-1]
                with torch.no_grad():
                    self.ca_layer.mlp.c_fc.weight.copy_(
                        self.decoder_layer.mlp.c_fc.weight.data.t()
                    )
                assert self.ca_layer.mlp.c_proj.weight.shape == self.decoder_layer.mlp.c_proj.weight.shape[::-1]
                with torch.no_grad():
                    self.ca_layer.mlp.c_proj.weight.copy_(
                        self.decoder_layer.mlp.c_proj.weight.data.t()
                    )
            else:
                # llama flavor
                # Assuming all the weight attributes are PyTorch tensors or parameters
                self.ca_layer.mlp.gate_proj.weight.data.copy_(
                    self.decoder_layer.mlp.gate_proj.weight.data
                )

                self.ca_layer.mlp.up_proj.weight.data.copy_(
                    self.decoder_layer.mlp.up_proj.weight.data
                )

                self.ca_layer.mlp.down_proj.weight.data.copy_(
                    self.decoder_layer.mlp.down_proj.weight.data
                )
            
    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.av_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_av_x(self, av_x):
        self.av_x = av_x

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        
        # Cross attention
        if self.ca_layer is not None:
            if self.av_x is None:
                raise ValueError("av_x must be conditioned before forward pass")
            else:
                # _lang_x = lang_x[0]
                lang_x = self.ca_layer(
                    lang_x,
                    self.av_x,
                )
        return lang_x


class BNEmbedding(nn.Module):
    def __init__(self, n_bn_fusion=-1, d=1024):
        super().__init__()
        self.bn_embedding = nn.Parameter(
            torch.zeros(n_bn_fusion, d)
        ) # ()
        # init
        torch.nn.init.normal_(
            self.bn_embedding,
            mean=0.0, std=0.02
        )
    
    def forward(
            self,
            lang_x,
            attention_mask=None,
            **decoder_layer_kwargs
        ):
        # (n, D) --. (1, n, D)
        bn_x = self.bn_embedding.unsqueeze(0)
        # (1, n, D) --> (B,n,D)
        bn_x = bn_x.repeat(lang_x.size(0), 1, 1)
        # (B, L+n, D)
        lang_x = torch.cat(
            (lang_x, bn_x), dim=1
        )
        return lang_x


    
class msaLMEmbedding(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native decoder LMLayer.
    """

    def __init__(
        self, embedding, seqaug=None, maxlen=50
    ):
        super().__init__()
        self.embedding = embedding
        # self.positional = positional
        self.seqaug = seqaug
        
    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Normal embedding layer
        lang_x = self.embedding(
            lang_x, **decoder_layer_kwargs
        )
        
        # sqaug layer
        if self.seqaug is not None and self.training:
            lang_x = self.seqaug(lang_x)
        
        return lang_x

class msaLMPositional(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native positional encoding LMLayer.
    """

    def __init__(
        self, positional, n_fusion=8
    ):
        super().__init__()
        self.positional = positional
        self.n_fusion = n_fusion

    def forward(
        self,
        pos_id,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        
        # positional embeddings - gpt case
        _, t = pos_id.size()
        device = pos_id.device
        pos = torch.arange(
            0, t, dtype=torch.long, device=device
        )
        pos_emb = self.positional(pos)
        return pos_emb


class msaLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name="model.layers"):
        self.decoder_layers_attr_name = decoder_layers_attr_name
    
    def set_embedding_attr_name(self, embedding_attr_name="model.embed_tokens"):
        self.embedding_attr_name = embedding_attr_name

    def set_positional_attr_name(self, positional_attr_name="transformer.wpe"):
        self.positional_attr_name = positional_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _get_embedding_layers(self):
        return getattr_recursive(self, self.embedding_attr_name)

    def _get_positional_layers(self):
        return getattr_recursive(self, self.positional_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)
    
    def _set_embedding_layers(self, value):
        setattr_recursive(self, self.embedding_attr_name, value)

    def _set_positional_layers(self, value):
        setattr_recursive(self, self.positional_attr_name, value)

    def init_msalm(
        self,
        ca_list,
        config
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.msa_config = config
        self.set_embedding_attr_name(
            self.msa_config.get("embedding_attr_name", "model.embed_tokens")
        )
        self.set_decoder_layers_attr_name(
            self.msa_config.get("decoder_layers_attr_name", "model.layers")
        )
        self.lm_flavor = self.msa_config["mmgpt"].get("type", "llama")
        if "gpt" in self.lm_flavor:
            self.set_positional_attr_name()
        print(f"ca list is: {ca_list}")
        self.init_embedding_layers()
        if "gpt" in self.lm_flavor:
            self.init_positional_layers()
        self.init_msalm_layers(ca_list)
        self.initialized_msalm = True
        self._use_cached_av_x = False
        self.n_bn_fusion = self.msa_config.get("n_bn_fusion", -1)

    def init_msalm_layers(self, ca_list):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        old_decoder_blocks = self._get_decoder_layers()
        ca_layers = nn.ModuleList(
            [
                MMBlock(
                    self.msa_config["mmgpt"], layer_idx
                )
                for layer_idx in range(len(ca_list))
            ]
        )
        
        all_layers = []
        for idx in range(len(old_decoder_blocks)):
            print(f"Parsing decoder block: {idx}")
            if idx in ca_list:
                # get the corresponding index in ca_list
                ca_idx = ca_list.index(idx)
                all_layers.append(
                    msaLMLayer(
                        ca_layers[ca_idx],
                        old_decoder_blocks[idx],
                        self.msa_config["max_token_len"],
                        self.msa_config["mmgpt"].get("type", "llama")
                    )
                )
            else:
                all_layers.append(
                    msaLMLayer(None, old_decoder_blocks[idx])
                )

        self._set_decoder_layers(nn.ModuleList(all_layers))
    
    def init_embedding_layers(self):
        """
        Re initializes the Flamingo Layers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        # get pretrained embedding block
        embedding_block = self._get_embedding_layers()
        positional_block = None
        # if "gpt" in self.lm_flavor:
        #     positional_block = self._get_positional_layers()
        if self.msa_config.get("use_seqaug", False):
            # print("initializing SoftPerm")
            seqaug_layer = SoftPerm(
                p_apply=self.msa_config["mmgpt"]["p_apply"],
                p_feat=self.msa_config["mmgpt"]["p_perm"],
                n_bn=0,
                maxlen=self.msa_config["max_token_len"]
            )

        else:
            seqaug_layer = None
        
        new_embedding = msaLMEmbedding(
            embedding_block,
            seqaug_layer,
            self.msa_config["max_token_len"],
        )
        self._set_embedding_layers(nn.Sequential(new_embedding))
    
    def init_positional_layers(self):
        # get pretrained embedding block
        positional_block = self._get_positional_layers()
        
        new_positional = msaLMPositional(
            positional_block
        )
        self._set_positional_layers(nn.Sequential(new_positional))
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_msalm:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        # package arguments for the other parent's forward. since we don't know the order of the arguments,
        # make them all kwargs
        # if 'gpt' in self.lm_flavor:
        #     # gpt2
        #     b, _ = input_ids.size()
        #     zero_ids = torch.zeros(
        #         (b, self.n_bn_fusion),
        #         dtype=torch.long,
        #         device=input_ids.device
        #     )
        #     input_ids = torch.cat(
        #         (input_ids, zero_ids), dim=1
        #     )
        #     kwargs["input_ids"] = input_ids
        #     kwargs["output_attentions"] = False
        # else:
        #     kwargs["input_ids"] = input_ids
        #     kwargs["output_attnetions"] = False
        
        kwargs["input_ids"] = input_ids
        kwargs["output_attentions"] = False

        # # append 1s to attention_mask
        # if (attention_mask is not None) and (self.n_bn_fusion > 0):
        #     # Create a tensor of ones with shape (B, n)
        #     ones_mask = torch.ones(
        #         input_ids.size(0),
        #         self.n_bn_fusion,
        #         device=attention_mask.device,
        #         dtype=attention_mask.dtype
        #     )
        #     # Concatenate the original attention_mask with the ones_mask
        #     attention_mask = torch.cat(
        #         (attention_mask, ones_mask), dim=1
        #     )
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_av_x(None)

class TAFC_O1(nn.Module):
    def __init__(self, config):
        """
        Flamingo-like ablation where there are no learnable fusion tokens at all just XATTN
        
        Args:
            config (dict): configuration for all other functionalities
        """
        super(TAFC_O1, self).__init__()
        # useful args
        self.max_token_len = config.max_token_len
        self.modded_loss = config.get("modded_loss", False)
        self.lm_flavor = config['mmgpt'].get("type", 'llama')

        # ------ language model ----------
        if 'chinese' in config.lm:
            lm_hf_path = f"uer/{config.lm}"
        else:
            lm_hf_path = config.lm
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            lm_hf_path,
            model_max_length=config.max_token_len
        )
        if 'chinese' in config.lm:
            # due to BERT tokenizer has pad and no eos
            self.tokenizer.eos_token = self.tokenizer.pad_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if config.get("use_bf16", False):
            llm_config = AutoConfig.from_pretrained(lm_hf_path)
            llm_config.output_attentions = False  # Ensure this is set to False
            llm_config.output_hidden_states = True
            # self.lang_encoder = AutoModelForCausalLM.from_pretrained(
            #     llm_config
            # )
            # self.lang_encoder = AutoModelForCausalLM.from_config(llm_config)
            self.lang_encoder = AutoModelForCausalLM.from_pretrained(
                lm_hf_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                return_dict_in_generate=True,
                output_hidden_states=True,
                use_cache=False,
            )
        else:
            if 'gpt' in self.lm_flavor:
                self.lang_encoder = AutoModelForCausalLM.from_pretrained(
                    lm_hf_path,
                    local_files_only=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
            else:                
                self.lang_encoder = AutoModelForCausalLM.from_pretrained(
                    lm_hf_path,
                    local_files_only=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
        
        # add mixing utilities
        extend_instance(self.lang_encoder, msaLMMixin)

        # add cross attentions
        self.lang_encoder.init_msalm(
            config["mmgpt"]["mm_layer"],
            config,
        )

        ##############################################################################
        # original code here - before bn - uncomment afterwards
        ##########################################################################
        # self.W_task = nn.Linear(config.mmgpt.d_mm, 1)
        self.d_task = config["mmgpt"].get("d_out", 64)
        self.W_task = nn.Sequential(
            nn.Linear(config.mmgpt.d_mm + config["av_enc"]["d_enc_out"], self.d_task),
            nn.BatchNorm1d(self.d_task),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_task, 1)
        )
        self.W_text = nn.Linear(config.mmgpt.d_mm, 1)
        self.W_av = nn.Linear(config["av_enc"]["d_enc_out"], 1)
        
        
        # ------ audiovisual encoder ----------
        self.av_encoder = AV_Enc(config)
        if config["av_enc"].get("from_pretrained", False):
            print("----------------->>> Pretrained AudioVisual Encoder <<<<<----------------")
            self.av_encoder = self.av_encoder.from_pretrained(
                config["av_enc"]["path_to_pretrained"],
                config
            )
        else:
            print("From Scratch AudioVisual Encoder Initialization")

        # add layer normalization
        self.use_lnorm = config.get("use_lnorm", False)
        if self.use_lnorm:
            print("------------------ Adding LNorm ------------------------")
            d_av = config["av_enc"]["d_enc_out"]
            # self.RMS = nn.RMSNorm(d_av)
            self.LN = nn.LayerNorm(d_av)

        
    def forward(
        self,
        lang_x: torch.Tensor,
        audio_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass of Flamingo.

        Args:
            lang_x (torch.Tensor): Language input ids
                shape (B, max_len)
            av_x (torch.Tensor): audiovisual fused input
                shape (B, max_len, D)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        """
        assert (
            self.lang_encoder.initialized_msalm
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        av_x = self._encode_av(audio_x=audio_x, vision_x=vision_x)
        # print("calling _av_conditioning")
        self._av_conditioning(av_x)

        outputs = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            **kwargs
        )

        lm_logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]
        
         
            
        # last hidden state of the LM
        B, _, _ = last_hidden_states.size()
        with torch.no_grad():
            last_valid_indices = \
                            torch.sum(attention_mask, dim=1).long() - 1
        last_hidden_text = \
            last_hidden_states[
                torch.arange(B, device=last_hidden_states.device),
                last_valid_indices
            ]
        text_logits = self.W_text(last_hidden_text)
        # av output
        av_x = torch.mean(av_x, dim=1)
        av_logits = self.W_av(av_x)
        # global fusion output
        task_logits = self._task_map(
            torch.cat((last_hidden_text, av_x), dim=1)
        )
        
        return {
            "task_logits": task_logits,
            "text_logits": text_logits,
            "av_logits": av_logits,
            "bn_logits": None,
            "lm_logits": lm_logits,
        } 

            # return lm_logits, task_logits, av_logits, bn_logits, text_logits
    
    def _task_map(self, h_last):
        return self.W_task(h_last)

    def _av_conditioning(self, av_x):
        # clear previous conditioning
        self.lang_encoder.clear_conditioned_layers()

        # condition
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_av_x(av_x)
    
    def _encode_av(self, audio_x: torch.Tensor, vision_x: torch.Tensor):
        """
        Compute audiovisual latent representations and condition language model.
        Args:
            audio_x (torch.Tensor): Audio input
            vision_x (torch.Tensor): Vision input
                shape (B, L_{m}, D_{m})
        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        # get new conditioning
        av_x = self.av_encoder(audio_x, vision_x)
        if self.use_lnorm:
            av_x = self.LN(av_x)
        return av_x
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x