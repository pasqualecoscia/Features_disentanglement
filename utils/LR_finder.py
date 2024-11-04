import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import torch
import os

class LRFinder:
    def __init__(self, model, optimizer, criterion, device, save_path):
        
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device
        self.save_init_params_path = os.path.join(save_path, 'init_params.pt')

        torch.save(model.state_dict(), self.save_init_params_path)

    def range_test(self, iterator, end_lr = 10, num_iter = 100, 
                   smooth_f = 0.05, diverge_th = 5):
        
        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        
        iterator = IteratorWrapper(iterator)
        
        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            #update lr
            lr_scheduler.step()
            
            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
                
            if loss < best_loss:
                best_loss = loss

            losses.append(loss)
            
            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
                       
        #reset model to initial parameters
        self.model.load_state_dict(torch.load(self.save_init_params_path))
                    
        return lrs, losses

    def _train_batch(self, iterator):
        
        self.model.train()
        
        self.optimizer.zero_grad()
        
        x, y = iterator.get_batch()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred = self.model(x)
                
        loss = self.criterion(y_pred, y)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels, _ = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, _, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)

import matplotlib.pyplot as plt

def plot_lr_finder(lrs, losses, skip_start=5, skip_end=5, save_path=None):
    # Check if the folder exists, and create it if not
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')

    if save_path:
        plt.savefig(os.path.join(save_path, 'lr_finder.png'))
    else:
        plt.show() 