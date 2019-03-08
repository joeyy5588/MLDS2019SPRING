import numpy as np
import torch
import os
from torchvision.utils import make_grid
from base import BaseTrainer
from numpy.linalg import eig

class HessianTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(HessianTrainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.total_epoch = config['trainer']['epochs']

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        state = False
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss = self.loss(output, target)
            
            grads = self._grad_list(loss)
            zeros = torch.zeros_like(grads)
            grad_loss = self.loss(grads, zeros)

            if epoch == self.total_epoch and state == False:
                state = True
                min_ratio = self._min_ratio(loss)
            
            if epoch < 100:
                loss.backward()
            else:
                grad_loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        if epoch == self.total_epoch:
            log = {
                'loss': total_loss / len(self.data_loader),
                'metrics': (total_metrics / len(self.data_loader)).tolist(),
                'grad_norm': self._calc_gradnorm(),
                'min_ratio': min_ratio
            }
        else:
            log = {
                'loss': total_loss / len(self.data_loader),
                'metrics': (total_metrics / len(self.data_loader)).tolist(),
                'grad_norm': self._calc_gradnorm()
            }

        return log
    def _calc_gradnorm(self):
        grad_all = 0
        for p in self.model.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_all += grad
        grad_norm = grad_all ** 0.5
        return grad_norm

    def _grad_list(self, out):
        grads = []
        for p in self.model.parameters():
            g, = torch.autograd.grad(out, p, create_graph = True)
            g = g.flatten()
            grads.append(g)
        grad_list = torch.cat(grads)
        return grad_list
    
    def _min_ratio(self, out):
        grad_list = self._grad_list(out)
        hessian = []
        for i in range(len(grad_list)):
            hessian.append([])
            for p in self.model.parameters():
                q, = torch.autograd.grad(grad_list[i], p, create_graph = True)
                q = q.flatten()
                hessian[i] = np.append(hessian[i], q.cpu().data.numpy())
        hessian = np.array(hessian)
        w, v = eig(hessian)
        positive_count = np.sum([1 if i > 0 else 0 for i in w])
        min_ratio = positive_count / len(w)
        print(min_ratio)
        return min_ratio