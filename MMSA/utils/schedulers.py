from sched import scheduler
import torch
from functools import partial
from torch.optim.lr_scheduler import (_LRScheduler,
                                      LambdaLR,
                                      ReduceLROnPlateau)
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


def get_scheduler(optimizer, max_epochs, steps_per_epoch, warmup_steps):
    # scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     eta_min=1e-7,
    #     last_epoch=-1,
    #     T_max=(max_epochs+1) * steps_per_epoch - warmup_steps
    # )
    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        eta_min=1e-7,
        last_epoch=-1,
        T_0=(max_epochs+1) * steps_per_epoch - warmup_steps,
    )
    scheduler_warmup = LinearWarmupScheduler(
        optimizer,
        warmup_steps,
    )
    seq_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [scheduler_warmup, scheduler_steplr],
        milestones=[warmup_steps]
    )

    return seq_scheduler


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        return float(1)
    # return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with a steady learning rate set in the optimizer, after a warmup period 
    during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == "__main__":
    import torch
    model = torch.nn.Linear(10, 10)
    params_decay = [p for n, p in model.named_parameters() if n is not 'bias']    
    params_no_decay = [p for n, p in model.named_parameters() if n is 'bias']    
    param_groups = [
        {'params': params_decay,
        'weight_decay': 0.001,
        'lr': 0.01},
        {'params': params_no_decay,
        'weight_decay': 0.0,
        'lr': 0.01},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    # lin_scheduler = get_linear_schedule_with_warmup(optimizer, 50)
    # pl_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.9,
    #     verbose=True,
    #     patience=4
    # )
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     [lin_scheduler, pl_scheduler],
    #     milestones=[100],
    #     verbose=True
    # )
    ### test cosine annelaing with warmup
    # cos_scheduler, _ = get_scheduler(
    #     optimizer,
    #     max_epochs=20,
    #     steps_per_epoch=2,
    #     warmup_rate=0.2
    # )
    max_epochs = 20
    steps_per_epoch = 64
    # scheduler_steplr = LinearWarmupScheduler(
    #     optimizer, 0.1 * max_epochs * steps_per_epoch, 1e-8
    # )
    scheduler_steplr = get_scheduler(
        optimizer, max_epochs, steps_per_epoch
    )
    # scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     last_epoch=-1,
    #     T_max=max_epochs * steps_per_epoch
    # )
    for k in range(1, max_epochs*steps_per_epoch+1):
        scheduler_steplr.step()
        print(scheduler_steplr.get_last_lr())
    import pdb; pdb.set_trace()
    ##### test linear warmup scedule
    for epoch in range(120):
        if epoch < 50:
            lin_scheduler.step()
            print(lin_scheduler.get_last_lr())
        else:
            if epoch == 50:
                loss = 0.5
            else:
                loss = 1.5
            pl_scheduler.step(loss)
            # print(pl_scheduler.state_dict())



