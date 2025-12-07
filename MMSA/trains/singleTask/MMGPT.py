import logging
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random

from ...utils import MetricsTop, dict_to_str
from ...utils.schedulers import get_linear_schedule_with_warmup, get_scheduler

logger = logging.getLogger('MMSA')

# __all__ = ['MMSeq2Seq']


def exponential_decay(initial_value, final_value, total_epochs):
    # Calculate the base of the exponential function
    if final_value == 0:
        b = (1e-6 / initial_value) ** (1 / total_epochs)
    else:
        b = (final_value / initial_value) ** (1 / total_epochs)

    # Calculate the decayed value for each epoch
    decayed_values = [initial_value * (b ** epoch) for epoch in range(total_epochs + 1)]
    if final_value == 0:
        decayed_values[-1] = 0.0
    return decayed_values


class repel_term(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(repel_term, self).__init__()
        # self.repel_weight = repel_weight
        self.epsilon = epsilon

    def forward(self, alpha):
        all_alpha = alpha * alpha + self.epsilon
        # Compute the repelling penalty
        repel_penalty = 1.0 / all_alpha
        return repel_penalty[0]


class repel_term_plus(nn.Module):
    def __init__(self, epsilon):
        super(repel_term_plus, self).__init__()
        # self.repel_weight = repel_weight
        self.epsilon = epsilon

    def forward(self, alpha):
        all_alpha = self.epsilon - alpha
        # Compute the repelling penalty
        repel_penalty = torch.clamp(all_alpha[0], min=0)
        return repel_penalty


class barlow_twins(nn.Module):
    def __init__(self):
        super(barlow_twins, self).__init__()
        self.lam = 5e-3

    def off_diagonal(self, x):
        # return a flattened view of the
        # off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        bsz, _ = z1.shape()
        c = z1.T @ z2
        # divide by bsz
        c.div_(bsz)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lam * off_diag
        return loss


class MMGPT():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss(reduction='none') if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        if self.args.use_clm:
            print(f"----------------- + Using CLM loss")
            # ignore index for padded elements
            self.clm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.aux = False
        if self.args.get("distil", False):
            print(f"----------------- + Using MSE distil loss")
            self.distil_crit = nn.L1Loss()
            self.w_distil = self.args.get("w_distil", 0.0)
            self.distil_pat = self.args.get("distil_pat", -1)
        self.av_distil = False
        self.use_tf = self.args["av_enc"].get("transformator", False)
        self.layer_cond = self.args["av_enc"].get("layer_cond", False)
        if self.layer_cond:
            self.decay_av_w = self.args["av_enc"]["decay_av_weights"]
        if self.args.get("av_distil", False):
            print(f"----------------- + Using AV-MSE distil loss")
            self.distil_av_crit_x0 = nn.L1Loss()
            self.distil_av_crit_xL = nn.L1Loss()
            self.w_av_distil = self.args.get("w_av_distil", 0.0)
            self.distil_pat = self.args.get("distil_pat", -1)
            self.p_av_distil = self.args.get("p_av_distil", -1)
            self.av_distil = True
        # warmup schedule
        self.warmup_epochs = self.args.get("warmup_epochs", -1)
        # add max epochs variable for new scheduler
        # self.max_epochs = 100
        self.max_epochs = self.args.get("max_epochs", 50)
        self.pretrained_av_enc = \
            self.args["av_enc"].get("from_pretrained", False)
        self.finetune_av_enc = self.args["av_enc"].get("finetune", True)

        # multimodal parameters
        self.mmgpt = args["mmgpt"]
        self.tune_ffw = self.mmgpt.get("tune_ffw", True)
        self.use_lora = self.mmgpt.get("use_lora", False)
        # if self.args.warmup_epochs > 0:
        #     self.warmup_epochs = self.args.warmup_epochs
        # else:
        #     self.warmup_epochs = -1
        # metrics
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.feature_name_map = {
            'T': 'Feature_t',
            'A': 'Feature_a',
            'V': 'Feature_v',
        }
        # self.args['device'] = 'cpu'

    def do_train(self, model, dataloader, return_epoch_results=False):
        # optimizer configuration
        if self.args.get("gamma_decay", False):
            self.gamma = exponential_decay(
                self.args.gamma, self.args.gamma_low, self.args.gamma_decay_epochs)
        else:
            self.gamma = self.args.gamma

        self.args["dense_decay"] = False

        gpt_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # import pdb; pdb.set_trace()
        if self.tune_ffw:
            # tune the whole FFW layer
            mmgpt_params = list(model.Model.mm_decoder.transformer.h_mm.named_parameters())
        elif self.use_lora:
            # tune only the LoRa
            mmgpt_params = []
            for k, v in model.Model.mm_decoder.transformer.h_mm.named_parameters():
                if ("mlp" in k):
                    if "lora" in k:
                        mmgpt_params.append((k, v))
                    else:
                        v.requires_grad = False
                else:
                    mmgpt_params.append((k, v))
        else:
            # freeze the FFW block
            mmgpt_params = []
            for k, v in model.Model.mm_decoder.transformer.h_mm.named_parameters():
                if ("mlp" in k):
                    v.requires_grad = False
                else:
                    mmgpt_params.append((k, v))

        if self.finetune_av_enc:
            print("------------------------ +Tune AV Encoder" +
                  " ******************************************* ")
            av_params = list(model.Model.av_encoder.named_parameters())  # all av params
            # import pdb; pdb.set_trace()
            if self.aux:
                av_params = \
                    av_params + list(model.Model.av_dec.named_parameters()) \
                    + list(model.Model.av_clf.named_parameters())
            # av encoder params
            av_params_decay = \
                [p for n, p in av_params if not any(nd in n for nd in gpt_no_decay)]
            av_params_no_decay = \
                [p for n, p in av_params if any(nd in n for nd in gpt_no_decay)]
        else:
            print("------------------------ +Do not tune av_params")
            if self.aux:
                av_params = list(model.Model.av_dec.named_parameters()) \
                    + list(model.Model.av_clf.named_parameters())
                # av encoder params
                av_params_decay = \
                    [p for n, p in av_params if not any(nd in n for nd in gpt_no_decay)]
                av_params_no_decay = \
                    [p for n, p in av_params if any(nd in n for nd in gpt_no_decay)]
            else:
                av_params_decay = []
                av_params_no_decay = []
        # transformator
        tf_params_decay = []
        tf_params_no_decay = []
        if self.use_tf:
            print("------------------------ +Tune Tranformator Layers")
            tf_params = list(
                model.Model.tf.named_parameters()
            )
            # tf params
            tf_params_decay = \
                [p for n, p in tf_params if not any(nd in n for nd in gpt_no_decay)]
            tf_params_no_decay = \
                [p for n, p in tf_params if any(nd in n for nd in gpt_no_decay)]
        # layer embedding params
        lcond_params = []
        if self.layer_cond:
            print("------------------------ +Tune Layer Embedding Layers")
            lcond_params_names = list(
                model.Model.embedding.named_parameters()
            )
            lcond_params = [p for n, p in lcond_params_names]

        task_head_params = list(model.Model.mm_decoder.lm_task_head.named_parameters())
        # gpt params
        mmgpt_params_decay = [p for n, p in mmgpt_params if not any(nd in n for nd in gpt_no_decay)]
        mmgpt_params_no_decay = [p for n, p in mmgpt_params if any(nd in n for nd in gpt_no_decay)]
        # task heads
        task_head_params = [p for n, p in task_head_params]
        optimizer_grouped_parameters = [
            {'params': mmgpt_params_decay,
             'weight_decay': self.args.weight_decay_mmgpt,
             'lr': self.args.learning_rate_mmgpt},
            {'params': mmgpt_params_no_decay,
             'weight_decay': 0.0,
             'lr': self.args.learning_rate_mmgpt},
            {'params': av_params_decay,
             'weight_decay': self.args.weight_decay_av,
             'lr': self.args.learning_rate_av},
            {'params': av_params_no_decay,
             'weight_decay': 0.0,
             'lr': self.args.learning_rate_av},
            {'params': tf_params_decay,
             'weight_decay': self.args.weight_decay_av,
             'lr': self.args.learning_rate_av},
            {'params': tf_params_no_decay,
             'weight_decay': 0.0,
             'lr': self.args.learning_rate_av},
            {'params': task_head_params,
             'weight_decay': self.args.weight_decay_mmgpt,
             'lr': self.args.learning_rate_mmgpt},
            {'params': lcond_params,
             'weight_decay': self.args.weight_decay_mmgpt,
             'lr': self.args.learning_rate_mmgpt}
        ]
        # freeze GPT-only params
        # model.Model.mm_decoder.transformer.wte.requires_grad = False
        # model.Model.mm_decoder.transformer.wpe.requires_grad = False
        # model.Model.mm_decoder.transformer.drop.requires_grad = False
        # model.Model.mm_decoder.transformer.ln_f.requires_grad = False
        for param in model.Model.mm_decoder.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.Model.mm_decoder.transformer.wpe.parameters():
            param.requires_grad = False
        for param in model.Model.mm_decoder.transformer.drop.parameters():
            param.requires_grad = False
        for param in model.Model.mm_decoder.transformer.ln_f.parameters():
            param.requires_grad = False
        for param in model.Model.mm_decoder.transformer.h.parameters():
            param.requires_grad = False
        if not self.finetune_av_enc:
            print("------------------------ +Freeze av_params")
            for param in model.Model.av_encoder.parameters():
                param.requires_grad = False

        optimizer = optim.AdamW(optimizer_grouped_parameters)
        if self.warmup_epochs > 0:
            steps_per_epoch = int(len(dataloader["train"]) / self.args.update_epochs)
            warmup_steps = steps_per_epoch * self.warmup_epochs
            warm_scheduler = get_scheduler(
                optimizer,
                self.max_epochs,
                steps_per_epoch,
                warmup_steps,
            )
            print(f"Will be using warmup for {warmup_steps} steps")
            # warm_scheduler = \
            #     get_linear_schedule_with_warmup(optimizer, warmup_steps)

        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            # model.init_mmt()
            total_aux_loss = 0.0
            train_loss = 0.0
            lm_loss = 0.0
            barlow_loss = 0.0
            mse_loss = 0.0
            left_epochs = self.args.update_epochs
            # aug_mix_ratio = self.args.get('aug_mix_ratio', 0.0)
            # aug_mix_steps = 0
            with tqdm(dataloader['train']) as td:
                step_counter = 0
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = torch.stack(
                        batch_data['gpt_tokens_in']
                        ).to(self.args.device).permute(1, 0)
                    lm_text_tgt = torch.stack(
                        batch_data['gpt_tokens_tgt']
                        ).to(self.args.device).permute(1, 0)
                    lm_mask = torch.stack(
                        batch_data['gpt_tgt_mask']
                    ).to(self.args.device).permute(1, 0)
                    # 1 for valid positions and -1 for invalid
                    # Create binary mask
                    task_binary_mask = (lm_mask == 1).int()

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    # lm_logits: (B, L, |V|)
                    # task_logits: (B, L, 1)
                    if self.av_distil:
                        lm_logits, task_logits, av_logits = model(text, audio, vision)
                        if isinstance(av_logits, list):
                            if self.decay_av_w:
                                w_av = \
                                    torch.arange(
                                        1,
                                        len(av_logits) + 1,
                                        device=self.args.device
                                    )
                                w_av = w_av / (torch.sum(w_av) + 1e-5)
                                av_logits = torch.cat(av_logits, dim=1)
                                av_logits = w_av * av_logits
                            else:
                                # import pdb; pdb.set_trace()
                                av_logits = torch.stack(
                                    av_logits, dim=1
                                )
                                av_logits.squeeze_(2)
                                av_logits.T
                            av_logits = torch.mean(av_logits, dim=1)
                        else:
                            av_logits.squeeze_(1)
                    else:
                        lm_logits, task_logits = model(text, audio, vision)

                    # compute loss
                    B, L = text.size()
                    if self.args.dense:
                        # reweighted loss computation
                        if self.args.dense_uni:
                            uniform = \
                                torch.ones(B, L,
                                           device=self.args.device,
                                           requires_grad=False
                                           )
                            norm_factor = \
                                torch.sum(task_binary_mask, dim=1) + 1e-6
                            uniform = uniform / norm_factor.unsqueeze_(1)
                            dense_mask = uniform * task_binary_mask
                        elif self.args.dense_lin_decay:
                            # Linearly decaying weights for each batch (B, L)
                            linear_weights_batch = \
                                torch.linspace(1, 0, L, device=self.args.device).repeat(B, 1)
                            # Apply the binary mask
                            linear_weights_batch = linear_weights_batch * task_binary_mask
                            # Renormalize so that each row sums to 1
                            row_sums = linear_weights_batch.sum(dim=1, keepdim=True)
                            dense_mask = linear_weights_batch / row_sums
                        else:
                            raise KeyError("No dense loss weighting. Pls check config")
                        # requires reduction = 'none', and manual averaging over B
                        #  time dimension is already averaged via reweighting
                        task_logits = task_logits.squeeze(2)
                        expanded_labels = labels.expand_as(task_logits)
                        task_loss = self.criterion(task_logits, expanded_labels)

                        if self.args.reweight_last:
                            last_valid_indices = \
                                torch.sum(task_binary_mask, dim=1).long() - 1
                            if self.args.dense_decay:
                                if (epochs-1) >= len(self.lam):
                                    lam_all = self.lam[-1]
                                else:
                                    lam_all = self.lam[epochs-1]
                                dense_mask = dense_mask * lam_all
                            else:
                                dense_mask = dense_mask * self.args.lam
                            dense_mask[
                                torch.arange(B, device=self.args.device),
                                last_valid_indices,
                                ] = 1 #1 - self.args.lam
                        task_loss = torch.sum(task_loss * dense_mask) / B
                    else:
                        # calculate last token only (assumes non-empty text)
                        last_valid_indices = \
                            torch.sum(task_binary_mask, dim=1).long() - 1
                        # Gather the last non-zero logits using the corrected indices
                        last_non_zero_logits = \
                            task_logits[
                                torch.arange(B, device=self.args.device),
                                last_valid_indices
                            ]
                        # TODO: woll require expanded labels here too for metric calculation below
                        task_loss = self.criterion(last_non_zero_logits, labels)

                    loss = task_loss

                    if self.args.get('distil', False):
                        last_valid_indices = \
                                    torch.sum(task_binary_mask, dim=1).long() - 1
                        last_logits = task_logits[
                            torch.arange(B, device=self.args.device),
                            last_valid_indices
                        ]
                        first_logits = task_logits[:, 0]
                        # if self.p_av_distil <= random.random():
                        if (self.distil_pat + 1) < epochs:
                        #     if self.args.get("distil_dict", None):
                        #         if self.args.distil_dict.method == "uniform":
                        #             random_timestep = random.randint(
                        #                 a=self.args.distil_dict.u_low,
                        #                 b=self.args.distil_dict.u_high
                        #             )
                        #             first_logits = task_logits[:, random_timestep]
                        #         elif self.args.distil_dict.method == "mean":
                        #             first_timestep = self.args.distil_dict.u_low
                        #             last_timestep = self.args.distil_dict.u_high
                        #             first_logits = torch.mean(
                        #                 task_logits[:, first_timestep: last_timestep+1],
                        #                 dim=1
                        #             )
                        #         else:
                        #             raise KeyError("no such method")
                        #     else:
                        #         first_logits = task_logits[:, 0]
                            distil_loss = \
                                self.distil_crit(last_logits, first_logits)
                            loss += self.w_distil * distil_loss

                        if self.args.get('av_distil', False):
                            av_distil_loss_L = \
                                self.distil_av_crit_xL(last_logits, av_logits)
                            av_distil_loss_0 = \
                                self.distil_av_crit_x0(first_logits, av_logits)
                            loss += \
                                self.w_av_distil * (av_distil_loss_0 + av_distil_loss_L)

                    clm_loss = .0
                    if self.args.use_clm:
                        # compute lm loss only on non-masked tokens
                        B, L, V = lm_logits.shape
                        lm_logits = lm_logits.view(B*L, V)
                        lm_text_tgt[~(task_binary_mask.bool())] = -1  # ignore index
                        lm_text_tgt = lm_text_tgt.reshape(B*L)
                        # ignore_idx = -1
                        clm_loss = self.clm_criterion(lm_logits, lm_text_tgt)
                        w_gamma = self.gamma
                        # total_loss
                        loss += w_gamma * clm_loss

                    # backward
                    loss.backward()
                    # grad clip
                    if self.args.grad_clip != -1.0:
                        # TODO: mmgpt might not require grad clipping
                        nn.utils.clip_grad_value_(
                            [param for _, param in mmgpt_params if param.requires_grad],
                            self.args.grad_clip
                        )
                        if self.finetune_av_enc or self.aux:
                            nn.utils.clip_grad_value_(
                                [param for _, param in av_params if param.requires_grad],
                                self.args.grad_clip
                            )
                        nn.utils.clip_grad_value_(
                            [param for param in task_head_params if param.requires_grad],
                            self.args.grad_clip
                        )
                    # store results
                    train_loss += task_loss.item()
                    lm_loss += clm_loss.item()
                    y_pred.append(task_logits.cpu())
                    # put expanded_labels here
                    y_true.append(expanded_labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        ###########################################################################
                        ## Old scheduler implementation
                        ###########################################################################
                        # TODO: investigate whether the first update is performed with large lr or not
                        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
                        # if self.warmup_epochs >= epochs:
                        #     warm_scheduler.step()
                        ###########################################################################
                        warm_scheduler.step()

                        if step_counter <=3:
                            my_dict = optimizer.state_dict()
                            # for k in my_dict['param_groups']:
                            #     print(k['lr'])
                        step_counter += 1

                        left_epochs = self.args.update_epochs
                # trick for last batch update
                if not left_epochs:
                    # update
                    optimizer.step()
                    ###########################################################################
                    ## Old scheduler implementation
                    ###########################################################################
                    # if self.warmup_epochs >= epochs:
                    #     warm_scheduler.step()

                    ###########################################################################
                    ## NEW scheduler implementation
                    ###########################################################################
                    warm_scheduler.step()
            train_loss = train_loss / len(dataloader['train'])
            lm_loss = lm_loss / len(dataloader['train'])
            # mse_loss = mse_loss / len(dataloader['train'])
            # barlow_loss = barlow_loss / len(dataloader['train'])
            total_aux_loss = total_aux_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >>" \
                f" loss: {round(train_loss, 4)} {dict_to_str(train_results)}" \
                f" clm loss: {round(lm_loss, 4)}" \
                f" distill loss: {round(mse_loss, 4)}" \
                f" barlow loss: {round(barlow_loss, 4)}" \
                f" aux loss: {round(total_aux_loss, 4)}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # if self.warmup_epochs < epochs:
            #     scheduler.step(val_results['Loss'])
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None
            # if epochs - best_epoch >= 15:
            #     return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        print(f"Model alphas are")
        for n, p in model.Model.mm_decoder.transformer.h_mm.named_parameters():
            if "alpha" in n:
                print(f"{n} is: {p}")
        y_pred, y_true = [], []
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = torch.stack(
                        batch_data['gpt_tokens_in']
                        ).to(self.args.device).permute(1, 0)
                    lm_mask = torch.stack(
                        batch_data['gpt_tgt_mask']
                    ).to(self.args.device).permute(1, 0)
                    task_binary_mask = (lm_mask == 1).int()
                    last_valid_indices = \
                            torch.sum(task_binary_mask, dim=1).long() - 1

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    if self.aux or self.args.get("av_distil", False):
                        lm_logits, task_logits, _ = model(text, audio, vision)
                    else:
                        lm_logits, task_logits = model(text, audio, vision)

                    B, L = text.shape
                    # Gather the last non-zero logits using the corrected indices
                    last_non_zero_logits = \
                        task_logits[
                            torch.arange(B, device=self.args.device),
                            last_valid_indices
                        ]

                    # # TODO: remove this in the future
                    # if return_sample_results:
                    #     ids.extend(batch_data['id'])
                    #     for item in features.keys():
                    #         features[item].append(outputs[item].cpu().detach().numpy())
                    #     all_labels.extend(labels.cpu().detach().tolist())
                    #     preds = outputs["M"].cpu().detach().numpy()
                    #     # test_preds_i = np.argmax(preds, axis=1)
                    #     sample_results.extend(preds.squeeze())

                    # SOS: here we evaluate only on the last logit as in vanilla setups
                    # B, _ = labels.shape
                    loss = self.criterion(last_non_zero_logits.view(-1), labels.view(-1))
                    # import pdb; pdb.set_trace()
                    loss = torch.mean(loss)
                    eval_loss += loss.item()
                    y_pred.append(last_non_zero_logits.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results

    def do_robust(self, model, dataloader, mode="corr", p=0.3):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    # adding noise
                    if mode == "iid":
                        bsz = text.size(0)
                        l_text, l_audio, l_vision = \
                            text.size(1), audio.size(1), vision.size(1)
                        mask_text = \
                            torch.bernoulli(torch.full((bsz, l_text), 1-p)).to(text.device).unsqueeze_(2)
                        mask_audio = \
                            torch.bernoulli(torch.full((bsz, l_audio), 1-p)).to(text.device).unsqueeze_(2)
                        mask_vision = \
                            torch.bernoulli(torch.full((bsz, l_vision), 1-p)).to(text.device).unsqueeze_(2)
                        text = text * mask_text
                        audio = audio * mask_audio
                        vision = vision * mask_vision
                    elif mode == "corr":
                        bsz = text.size(0)
                        l_text, l_audio, l_vision = \
                            text.size(1), audio.size(1), vision.size(1)
                        max_len = max(l_text, l_audio, l_vision)
                        mask = \
                            torch.bernoulli(torch.full((bsz, max_len), 1-p)).to(text.device).unsqueeze_(2)
                        text = text * mask[:, :l_text]
                        audio = audio * mask[:, :l_audio]
                        vision = vision * mask[:, :l_vision]
                    else:
                        raise KeyError(f"Not a valid noise type")

                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(text, audio, vision)

                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"Robust Eval: {mode}-{p}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        return eval_results


    def do_dominance(self, model, dataloader, mode="zero", p=0.3, train_dataloader=None):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        if mode == "mean_text" or mode == "mean_av":
            # Initialize variables to accumulate the sum and count for each modality
            total_vision_sum = \
                torch.zeros_like(next(iter(train_dataloader))['vision'][0]).to(self.args.device)
            total_audio_sum = \
                torch.zeros_like(next(iter(train_dataloader))['audio'][0]).to(self.args.device)
            total_text_sum = \
                torch.zeros_like(next(iter(train_dataloader))['text'][0]).to(self.args.device)
            total_count = 0
            with torch.no_grad():
                with tqdm(train_dataloader) as td:
                    # Iterate through the DataLoader
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        batch_size = vision.size(0)  # Get the actual batch size
                        # Accumulate the sums for each modality
                        total_vision_sum += vision.sum(dim=0)
                        total_audio_sum += audio.sum(dim=0)
                        total_text_sum += text.sum(dim=0)

                        # Accumulate the count
                        total_count += batch_size

                # Calculate the mean representations for each modality
                mean_vision = total_vision_sum / total_count
                mean_audio = total_audio_sum / total_count
                mean_text = total_text_sum / total_count

                # Reshape to (1, L, D) for each modality
                mean_vision = mean_vision.unsqueeze(0)
                mean_audio = mean_audio.unsqueeze(0)
                mean_text = mean_text.unsqueeze(0)

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    # adding noise
                    if mode == "mean_text":
                        if p > random.random():
                            text = mean_text.expand_as(text)
                    elif mode == "mean_av":
                        if p > random.random():
                            audio = mean_audio.expand_as(audio)
                            vision = mean_vision.expand_as(vision)
                    elif mode == "zero_text":
                        if p > random.random():
                            text = text * 0.0
                    elif mode == "zero_av":
                        if p > random.random():
                            audio = audio * 0.0
                            vision = vision * 0.0
                    else:
                        raise KeyError(f"Not a valid noise type")

                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(text, audio, vision)

                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"Dominance Eval: {mode}-{p}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        return eval_results

    def do_test_head(self, backbone, model, dataloader, mode="VAL", modal="T"):
        backbone.eval()
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    last_h = backbone(text, audio, vision)[self.feature_name_map[modal]]
                    outputs = model(last_h)

                    loss = self.criterion(outputs, labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}-{modal}) >> {dict_to_str(eval_results)}")

        return eval_results
