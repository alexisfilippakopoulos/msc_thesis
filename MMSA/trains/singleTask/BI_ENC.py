import logging
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import (ReduceLROnPlateau,
                                      CosineAnnealingWarmRestarts,
                                      SequentialLR, _LRScheduler)
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, initial_lr=None, last_epoch=-1, verbose=False):
        # target_lr is derived from each parameter group in the optimizer
        self.target_lr = [group['lr'] for group in optimizer.param_groups]

        # If initial_lr is provided, use it; otherwise, default to 0.0 for each group
        if initial_lr is not None:
            self.initial_lr = [initial_lr for _ in self.target_lr]
        else:
            self.initial_lr = [0.0 for _ in self.target_lr]

        self.warmup_steps = warmup_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Compute the learning rate for each parameter group
            lr_increment = [(target - initial) / self.warmup_steps for target, initial in zip(self.target_lr, self.initial_lr)]
            return [initial + increment * self.last_epoch for initial, increment in zip(self.initial_lr, lr_increment)]
        else:
            # Warmup complete, use target learning rates
            return self.target_lr

    def step(self, epoch=None):
        # Overriding step to allow updates per batch (or "step") instead of per epoch
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Update the learning rate for each parameter group and keep track of the new rates
        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr

        # Update the _last_lr attribute with the new learning rates
        self._last_lr = new_lrs


class BI_ENC():
    def __init__(self, args):
        self.args = args
        self.scheduler_cfg = args.get("scheduler", False)
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate)
        if self.scheduler_cfg is False:
            print(f"------- Ongoing with the ReduceLROnPlateau -----------")
            scheduler = ReduceLROnPlateau(optimizer,
                                        mode='min',
                                        factor=0.1,
                                        verbose=True,
                                        patience=self.args.patience)
        else:
            print(f"------- Ongoing with the LR with Warmup -----------")
            steps_per_epoch = int(len(dataloader['train']) / self.args.update_epochs)
            warmup_steps = steps_per_epoch * self.args['scheduler']['warmup_epochs']
            # warmup_steps = len(dataloader) * self.args['scheduler']['warmup_epochs']
            if self.args['scheduler'].get('T0', 0) > 0:
                restart_T = (self.args['scheduler']['T0']) * steps_per_epoch
                print(f"----> using restart period = {restart_T}")
            else:
                restart_T = (self.args['max_epochs']+1) * len(dataloader['train']) - warmup_steps
            scheduler_steplr = CosineAnnealingWarmRestarts(
                optimizer,
                eta_min=1e-7,
                last_epoch=-1,
                T_0=restart_T,
            )
            scheduler_warmup = LinearWarmupScheduler(
                optimizer,
                warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                [scheduler_warmup, scheduler_steplr],
                milestones=[warmup_steps]
            )
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
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    # grad accumulation
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    # print(f"---- vision shape is {vision.shape} ----")
                    # print(f"---- audio shape is {audio.shape} ----")
                    # import pdb; pdb.set_trace()
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    outputs = model(None, audio, vision)
                    # compute loss
                    loss = self.criterion(outputs, labels)
                    # compute gradients
                    loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        if self.scheduler_cfg is not False:
                            scheduler.step()
                            # print(scheduler.get_last_lr())
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            if self.scheduler_cfg is False:
                scheduler.step(val_results['Loss'])
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                print(f"***********************************************Saving Model at {self.args.model_save_path}")
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

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
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
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(None, audio, vision)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs.cpu().detach().numpy()
                        # test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(preds.squeeze())

                    loss = self.criterion(outputs, labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
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