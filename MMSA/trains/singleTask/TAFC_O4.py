import os
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.cuda.amp as amp

from tqdm import tqdm
from torch import optim
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau


from ...utils import MetricsTop, dict_to_str
from ...utils.schedulers import get_linear_schedule_with_warmup, get_scheduler

logger = logging.getLogger('MMSA')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# __all__ = ['MMSeq2Seq']


class TAFC_O4():
    def __init__(self, args):
        self.args = args
        # bf16 - training
        self.use_bf16 = self.args.get("use_bf16", False)
        if self.use_bf16:
            self.scaler = amp.GradScaler()

        # lm-flavor
        self.lm_flavor = self.args["mmgpt"].get("type", "llama")

        # modified loss fot bn version
        self.modded_loss = self.args.get("modded_loss", False)
        self.use_cmc_loss = self.args.get("av_distil", False)
        self.use_clm_loss = self.args.get("use_clm", False)
        self.n_bn_fusion = self.args.get("n_bn_fusion", -1)

        self.criterion = nn.L1Loss(reduction='none') if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        # extra losses
        if self.modded_loss:
            # self.crit_text = nn.L1Loss(reduction='none')
            self.crit_bn = nn.L1Loss(reduction='none')
            self.crit_av = nn.L1Loss(reduction='none')
        # av distil loss
        if self.use_cmc_loss:
            self.crit_cmc = nn.L1Loss(reduction='none')
        if self.use_clm_loss:
            self.clm_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # warmup schedule
        self.warmup_epochs = self.args.get("warmup_epochs", -1)
        
        # add max epochs variable for new scheduler
        self.max_epochs = self.args.get("max_epochs", 50)
        self.pretrained_av_enc = \
            self.args["av_enc"].get("from_pretrained", False)
        self.finetune_av_enc = self.args["av_enc"].get("finetune", True)

        # multimodal parameters
        self.mmgpt = args["mmgpt"]
        self.tune_ffw = self.mmgpt.get("tune_ffw", True)
        self.use_lora = self.mmgpt.get("use_lora", False)
        
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
        self.gamma = self.args.gamma
        self.args["dense_decay"] = False

        gpt_no_decay = [
            'bias',
            'LayerNorm.bias',
            'LayerNorm.weight',
            '_LlamaRMSNorm.weight'
        ]

        # Freeze all parameters
        model.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

        # tune the cross attention layers
        if "gpt" in self.lm_flavor:
            trainable_list = [
                "alpha_1", "alpha_2", "ln_1", "ln_2", "attn",
                "bn_embedding"
            ]
            if self.mmgpt.use_lora:
                trainable_list.extend(
                    ["lora_c_fc", "lora_c_proj"]
                )
            else:
                # full fine-tuning
                trainable_list.extend(
                    ["c_fc", "c_proj"]
                )                     
        else:
            trainable_list = [
                "alpha_1", "alpha_2", "ln_1", "ln_2", "attn",
                "lora_gate_proj", "lora_up_proj", "lora_down_proj",
                "bn_embedding"
            ]

        if "gpt" in self.lm_flavor:
            # gpt
            for n, p in model.Model.lang_encoder.transformer.h.named_parameters():
                if any(s in n for s in trainable_list) and ("ca_layer" in n):
                    print(n)
                    p.requires_grad_(True)
            for n, p in model.Model.lang_encoder.transformer.wte.named_parameters():
                if any(s in n for s in trainable_list):
                    print(n)
                    p.requires_grad_(True)
        else:
            # llama
            for n, p in model.Model.lang_encoder.model.layers.named_parameters():
                if any(s in n for s in trainable_list) and ("ca_layer" in n):
                    print(n)
                    p.requires_grad_(True)
            for n, p in model.Model.lang_encoder.model.embed_tokens.named_parameters():
                if any(s in n for s in trainable_list):
                    print(n)
                    p.requires_grad_(True)

        # AV encoder tuning 
        model.Model.av_encoder.requires_grad_(True)
        # Task layer tuning
        model.Model.W_task.requires_grad_(True)
        model.Model.W_bn.requires_grad_(True)
        # model.Model.W_text.requires_grad_(True)
        model.Model.W_av.requires_grad_(True)
        if self.args.use_lnorm:
            model.Model.LN.requires_grad_(True)
        # model.Model.av_dec.requires_grad_(True)

        total_trainable = 0
        for n, p in model.named_parameters():
            if  p.requires_grad:
                print(n)
                total_trainable += p.numel()
        # Convert to millions and format the output
        total_trainable_millions = total_trainable / 1_000_000
        print(f"The total number of trainable parameters is {total_trainable_millions:.2f} M")
        # print(f"The totalnumber of trainable parameters is {total_trainable}")
        # kke = model.Model.lang_encoder.model.layers[23]
        # a = kke.ca_layer.mlp.gate_proj.weight
        # b = kke.decoder_layer.mlp.gate_proj.weight
        # import pdb;pdb.set_trace()

        # seperate av params w and wout decay
        av_params = list(model.Model.av_encoder.named_parameters())  # all av params
        # av encoder params
        av_params_list = [f'Model.av_encoder.{n}' for n,_ in av_params]
        av_params_decay = \
            [p for n, p in av_params if not any(nd in n for nd in gpt_no_decay)]
        av_params_no_decay = \
            [p for n, p in av_params if any(nd in n for nd in gpt_no_decay)]

        # separate to params w and wout decay
        params_decay = []
        params_no_decay = []
        for n, p in model.named_parameters():
            # check if already in av_params_list
            if n not in av_params_list:
                print(n)
                if p.requires_grad:
                    if any(nd in n for nd in gpt_no_decay):
                        print(f"Using grad with no decay in {n}")
                        params_no_decay.append(p)
                    else:
                        print(f"Using grad with decay in {n}")
                        params_decay.append(p)
        
        # check if av_params + params are equal to total_trainable
        
        # Calculate the total number of elements
        tot_av_params_decay = sum(tensor.numel() for tensor in av_params_decay)
        tot_av_params_no_decay = sum(tensor.numel() for tensor in av_params_no_decay)
        tot_params_decay = sum(tensor.numel() for tensor in params_decay)
        tot_params_no_decay = sum(tensor.numel() for tensor in params_no_decay)
        assert total_trainable == (
                    tot_av_params_decay +
                    tot_av_params_no_decay +
                    tot_params_decay +
                    tot_params_no_decay
        )
        
        optimizer_grouped_parameters = [
            {
                'params': params_decay,
                'weight_decay': self.args.weight_decay_mmgpt,
                'lr': self.args.learning_rate_mmgpt,
                'betas': (
                    self.args.get('beta_1', 0.9),
                    self.args.get('beta_2', 0.999)
                )
            },
            {
                'params': params_no_decay,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate_mmgpt,
                'betas': (
                    self.args.get('beta_1', 0.9),
                    self.args.get('beta_2', 0.999)
                )

            },
            {
                'params': av_params_decay,
                'weight_decay': self.args.weight_decay_av,
                'lr': self.args.learning_rate_av,
                'betas': (
                    self.args.get('beta_1', 0.9),
                    self.args.get('beta_2', 0.999)
                )
            },
            {
                'params': av_params_no_decay,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate_av,
                'betas': (
                    self.args.get('beta_1', 0.9),
                    self.args.get('beta_2', 0.999)
                )

            }
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters)

        ###########################################################################################
        ## new version of sceduler
        ###########################################################################################
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
        min_or_max = 'min' if self.args.KeyEval in ['Loss', 'MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        ###########################################################################################
        ## training loop
        ###########################################################################################
        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            # model.init_mmt()
            total_loss = 0.0
            total_aux_loss = 0.0
            train_loss = 0.0
            lm_loss = 0.0
            barlow_loss = 0.0
            mse_loss = 0.0
            bn_total_loss = .0
            av_total_loss = .0
            text_total_loss = .0
            # loss = .0
            left_epochs = self.args.update_epochs
            ids = []
            # aug_mix_ratio = self.args.get('aug_mix_ratio', 0.0)
            # aug_mix_steps = 0
            # print(model.Model.lang_encoder.model.embed_tokens[0].bn_embedding.bn_embedding)
            with tqdm(dataloader['train']) as td:
                step_counter = 0
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    
                    # idx-es for ulgm
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)
                    
                    # language modality handling
                    raw_text = batch_data['raw_text']
                    tokenized_inputs = model.Model.tokenizer(
                        raw_text, return_tensors="pt",
                        padding_side='right',
                        padding="max_length",
                        truncation=True
                    ).to(self.args.device)
                    text_ids = tokenized_inputs['input_ids']
                    
                    # 1 for valid positions and -1 for invalid --- Create binary mask
                    attention_mask = tokenized_inputs['attention_mask']
                    # prepa lm ids
                    # Shift the tensor to the left by 1
                    lm_text_tgt = torch.roll(text_ids, -1, dims=1)
                    # Replace the last element of each row with eos_token
                    lm_text_tgt[:, -1] = 0
                    
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    # lm_logits: (B, L, |V|)
                    # task_logits: (B, L, 1)
                    if self.use_bf16:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                           outputs = model(
                               text_ids,
                               audio,
                               vision,
                               attention_mask=attention_mask
                            )
                           #lm_logits = outputs['lm_logits']
                           task_logits = outputs['task_logits']
                           av_logits = outputs['av_logits']
                           bn_logits = outputs['bn_logits']
                           #text_logits = outputs['text_logits']
                    else:
                        outputs = model(text_ids, audio, vision, attention_mask=attention_mask)
                        # lm_logits = outputs['lm_logits']
                        task_logits = outputs['task_logits']
                        av_logits = outputs['av_logits']
                        bn_logits = outputs['bn_logits']
                        # text_logits = outputs['text_logits']
                    av_logits.squeeze_(1)

                    # compute loss
                    loss = .0
                    B, L = text_ids.size()
                    if self.n_bn_fusion > 0 and self.modded_loss:
                        #######################################################################
                        ## mutlimodal task loss calculation
                        expanded_labels = labels.expand_as(task_logits)
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                task_loss = self.criterion(task_logits, expanded_labels)
                                task_loss = torch.mean(task_loss) # average over mini-batch
                        else:
                            task_loss = self.criterion(task_logits, expanded_labels)
                            task_loss = torch.mean(task_loss) # average over mini-batch
                        #######################################################################
                        ## bn fusion loss
                        bn_labels = labels.unsqueeze(1)
                        bn_labels = bn_labels.expand_as(bn_logits)
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                bn_loss = self.crit_bn(bn_logits, bn_labels)
                                bn_loss = torch.mean(bn_loss) # average over fusion tokens and mini-batch
                        else:
                            bn_loss = self.crit_bn(bn_logits, bn_labels)
                            # here we can manipulate each pf the `n_bn_fusion` tokens differently if we wish
                            bn_loss = torch.mean(bn_loss) # average over fusion tokens and mini-batch
                        #######################################################################
                        ## av loss
                        av_logits = av_logits.unsqueeze(1)
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                av_loss = self.crit_av(av_logits, labels)
                                av_loss = torch.mean(av_loss)
                        else:
                            av_loss = self.crit_av(av_logits, labels)
                            # here we can manipulate each pf the `n_bn_fusion` tokens differently if we wish
                            av_loss = torch.mean(av_loss) # average over fusion tokens and mini-batch
                    
                    # distil loss
                    if (self.n_bn_fusion > 0) and self.modded_loss:
                        if self.use_bf16:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                # L = global_loss + BN_loss + AV_loss + Lang_loss
                                loss = \
                                    task_loss \
                                    + self.args.l_bn * bn_loss \
                                    + self.args.l_av * av_loss
                        else:
                            loss = \
                                task_loss \
                                + self.args.l_bn * bn_loss \
                                + self.args.l_av * av_loss
                    else:
                        # loss here is the "ensemble" and weighted from all involved 
                        # timesteps in each method
                        loss = task_loss

                        if self.args.n_bn_fusion > 0:
                            first_logits = task_logits[:, 0]
                            last_logits = task_logits[:, -1]
                        else:
                            last_valid_indices = \
                                        torch.sum(attention_mask, dim=1).long() - 1
                            last_logits = task_logits[
                                torch.arange(B, device=self.args.device),
                                last_valid_indices
                            ]
                            first_logits = task_logits[:, 0]
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): 
                                distil_loss = \
                                    self.distil_crit(last_logits, first_logits)
                                loss += self.w_distil * distil_loss
                        else:
                            distil_loss = \
                                self.distil_crit(last_logits, first_logits)
                            loss += self.w_distil * distil_loss

                    if self.use_bf16:
                        self.scaler.scale(loss).backward()
                    else:
                        # backward
                        loss.backward()
                    
                    # # grad clip
                    # if self.args.grad_clip != -1.0:
                    #     if self.use_bf16:
                    #         # Unscales the gradients of optimizer's assigned params in-place
                    #         self.scaler.unscale_(optimizer)
                    #     # TODO: mmgpt might not require grad clipping
                    #     nn.utils.clip_grad_value_(
                    #         [param for param in params_decay],
                    #         self.args.grad_clip
                    #     )
                    #     nn.utils.clip_grad_value_(
                    #         [param for param in params_no_decay],
                    #         self.args.grad_clip
                    #     )

                    # store results
                    total_loss += loss.item()
                    train_loss += task_loss.item()
                    if self.modded_loss:
                        bn_total_loss += bn_loss.item()
                        av_total_loss += av_loss.item()
                    y_pred.append(task_logits.to(torch.float32).cpu().detach())
                    # put expanded_labels here
                    y_true.append(expanded_labels.to(torch.float32).cpu().detach())
                    
                    if not left_epochs:
                        if self.use_bf16:
                             # grad clip
                            if self.args.grad_clip != -1.0:
                                # Unscales the gradients of optimizer's assigned params in-place
                                self.scaler.unscale_(optimizer)
                                    # TODO: mmgpt might not require grad clipping
                                nn.utils.clip_grad_value_(
                                    [param for param in params_decay],
                                    self.args.grad_clip
                                )
                                nn.utils.clip_grad_value_(
                                    [param for param in params_no_decay],
                                    self.args.grad_clip
                                )
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            # grad clip
                            if self.args.grad_clip != -1.0:
                            # TODO: mmgpt might not require grad clipping
                                nn.utils.clip_grad_value_(
                                    [param for param in params_decay],
                                    self.args.grad_clip
                                )
                                nn.utils.clip_grad_value_(
                                    [param for param in params_no_decay],
                                    self.args.grad_clip
                                )
                            optimizer.step()
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
                    if self.use_bf16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()  
                    # optimizer.step()
                    warm_scheduler.step()
            
            train_loss = train_loss / len(dataloader['train'])
            total_loss = total_loss / len(dataloader['train'])
            bn_total_loss = bn_total_loss / len(dataloader['train'])
            av_total_loss = av_total_loss / len(dataloader['train'])
            total_aux_loss = total_aux_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >>" \
                f" loss: {round(train_loss, 4)} {dict_to_str(train_results)}" \
                f" total loss: {round(total_loss, 4)}" \
                f" bn loss: {round(bn_total_loss, 4)}" \
                f" av loss: {round(av_total_loss, 4)}" \
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
        if 'gpt' in self.lm_flavor:
            for n, p in model.Model.lang_encoder.transformer.h.named_parameters():
                if "alpha" in n:
                    print(f"{n} is: {p}")
        else:
            for n, p in model.Model.lang_encoder.model.layers.named_parameters():
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
                    # language modality handling
                    raw_text = batch_data['raw_text']
                    tokenized_inputs = model.Model.tokenizer(
                        raw_text, return_tensors="pt",
                        padding_side='right',
                        padding="max_length",
                        truncation=True
                    ).to(self.args.device)
                    text_ids = tokenized_inputs['input_ids']
                    attention_mask = tokenized_inputs['attention_mask']

                    # task_binary_mask = (attention_mask == 1).int()
                    last_valid_indices = \
                            torch.sum(attention_mask, dim=1).long() - 1

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    
                    if self.modded_loss:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = model(text_ids, audio, vision, attention_mask=attention_mask)
                            task_logits = outputs['task_logits']
                            # av_logits = outputs['av_logits']
                            # av_logits.squeeze_(1)                        
                    elif self.aux or self.args.get("av_distil", False):
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                lm_logits, task_logits, _, bn_logits = \
                                    model(
                                        text_ids,
                                        audio,
                                        vision,
                                        attention_mask=attention_mask
                                    )
                        else:
                            lm_logits, task_logits, _, _ = \
                                model(
                                    text_ids,
                                    audio,
                                    vision,
                                    attention_mask=attention_mask
                                )
                    else:
                        lm_logits, task_logits = model(text_ids, audio, vision)

                    B, L = text_ids.shape
                    
                    if self.modded_loss:
                        ##### task loss
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                loss = self.criterion(task_logits.view(-1), labels.view(-1))
                                loss = torch.mean(loss)
                        else:
                            loss = self.criterion(task_logits.view(-1), labels.view(-1))
                            loss = torch.mean(loss)
                    eval_loss += loss.item()
                    
                    # gather predictions
                    if self.modded_loss:
                        y_pred.append(task_logits.to(torch.float32).cpu().detach())
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

    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion', epochs=100):
        if mode == 'bn':
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.squeeze(2)
            y_true = y_true.expand_as(y_pred)
        else:
            y_pred = y_pred.view(-1)
            y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            if epochs > self.args.update_labels_patience + 1:
                weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
                if mode == "bn":
                    weighted = weighted.unsqueeze(1)
                # print(f"For {mode} the avg weight is: {torch.mean(weighted)}")
            else:
                if mode == "bn":
                    weighted = self.args.l_bn
                elif mode == "av":
                    weighted = self.args.l_av
                else:
                    weighted = self.args.l_t
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss
    
    def update_features(self, f_fusion, f_text, f_av, f_bn, indexes):
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text
        self.feature_map['av'][indexes] = f_av
        self.feature_map['bn'][indexes] = f_bn

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
        update_single_center(mode='text')
        update_single_center(mode='av')
        update_single_center(mode='bn')
    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['av'][indexes] = m_labels
        self.label_map['bn'][indexes] = m_labels
    
    def update_labels(self, f_fusion, f_text, f_av, f_bn, cur_epoches, indexes, outputs):
        MIN = 1e-8
        def update_single_label(f_single, mode):
            d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1) 
            d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1) 
            delta_s = (d_sn - d_sp) / (d_sp + MIN)
            # d_s_pn = torch.norm(self.center_map[mode]['pos'] - self.center_map[mode]['neg'], dim=-1)
            # delta_s = (d_sn - d_sp) / (d_s_pn + MIN)
            alpha = delta_s / (delta_f + MIN)

            new_labels = 0.5 * alpha * self.label_map['fusion'][indexes] + \
                        0.5 * (self.label_map['fusion'][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
            # new_labels = torch.tanh(new_labels) * self.args.H

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        d_fp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1) 
        d_fn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1) 
        # d_f_pn = torch.norm(self.center_map['fusion']['pos'] - self.center_map['fusion']['neg'], dim=-1)
        # delta_f = (d_fn - d_fp) / (d_f_pn + MIN)
        delta_f = (d_fn - d_fp) / (d_fp + MIN)
        
        update_single_label(f_text, mode='text')
        update_single_label(f_av, mode='av')
        update_single_label(f_bn, mode='bn')
