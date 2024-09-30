from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):

    def __init__(self, optimizer, initial_lr: float, max_steps: int, num_of_cycles:int = 1, gamma: float = 0.8, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps #$ total amount of epochs
        self.exponent = exponent #$ epsilon
        self.num_of_cycles = num_of_cycles #$ amount of cycles C
        self.ctr = 0
        self.gamma = gamma #$ gamma sets the start of epoch part two
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        #$ if number of cycles is 1, then we use the regular nnUNet poly lr scheduler
        if self.num_of_cycles == 1:
            if current_step is None or current_step == -1:
                current_step = self.ctr
                self.ctr += 1

            new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        #$ If number of cycles is more than 1, then we use the modified poly lr scheduler
        #$ alpha_r is rigid. It is set to 0.01 - this is the initial lr for the first step of each cycle.
        else:
            if current_step is None or current_step == -1:
                current_step = self.ctr
                self.ctr += 1
            
            Tc = self.max_steps//self.num_of_cycles
            tc = current_step%Tc
            step_part = min(tc,int(self.gamma*Tc))
            print('->>>>>>>>>>  current tc is:' ,tc)
            if tc == 0 :
                new_lr = self.initial_lr
            else:
                alpha_r = 0.01
                new_lr = alpha_r * (1 - step_part / self.max_steps) ** self.exponent

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr