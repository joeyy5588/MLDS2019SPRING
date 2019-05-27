import numpy as np
import torch
import torch.nn.functional as F
import logging
import os
import random
import torch.nn as nn
import math
from torchvision.utils import save_image
import torch.autograd as autograd


class WGANGPTrainer:    
    def __init__(self, gen, dis, dataloader, opt):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.n_epochs = opt.n_epochs
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.noise_dim = 100
        self.device = self._prepare_gpu()
        self.gen = gen
        self.dis = dis
        self.gen_iter = 1
        self.dis_iter = 1
        self.gen_optimizer = torch.optim.RMSprop(self.gen.parameters(), lr = 0.00005)
        self.dis_optimizer = torch.optim.RMSprop(self.dis.parameters(), lr = 0.00005)
        self.fixed_noise = torch.randn(25, self.noise_dim, 1, 1, device = self.device)
        self.criterion = nn.BCELoss()
        self.real_label = 1
        self.fake_label = 0
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.checkpoint)

    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.gen.to(self.device)
        self.dis.to(self.device)
        self.logger.info('[GEN_STRUCTURE]')
        self.logger.info(self.gen)
        self.logger.info('[DIS_STRUCTURE]')
        self.logger.info(self.dis)

        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log = self._train_epoch(i)
            merged_log = {**log}
            all_log.append(merged_log)
            checkpoint = {
                'log': all_log,
                'gen_state_dict': self.gen.state_dict(),
                'dis_state_dict': self.dis.state_dict(),
            }
            if (i+1)%5 == 0:
                check_path = os.path.join(opt.save_dir, 'wgan_checkpoint_' + str(i+1) + '.pth')
                torch.save(checkpoint, check_path)

    def _train_epoch(self, epoch):
        self.gen.train()
        self.dis.train()
        G_sum_loss = 0
        D_sum_loss = 0

        for batch_idx, real_images  in enumerate(self.dataloader):

            #Train Discriminator
            self.dis_optimizer.zero_grad()
            real_images = real_images.to(self.device)
            input_noise = torch.randn(real_images.size()[0], self.noise_dim, 1, 1).to(self.device)
            fake_images = self.gen(input_noise).detach()
            alpha = torch.randn(real_images.size()[0], 1, 1, 1).to(self.device)
            interpolate_images = (alpha * real_images + ((1 - alpha) * fake_images)).to(self.device).requires_grad_(True)
            real_label = torch.ones(real_images.size()[0]).to(self.device)
            fake_label = torch.zeros(real_images.size()[0]).to(self.device)
            real_predict = self.dis(real_images)#.view(-1)
            fake_predict = self.dis(fake_images)#.view(-1)
            interpolate_predict = self.dis(interpolate_images)
            real_loss = real_predict.mean()
            fake_loss = fake_predict.mean()
            interpolate_loss = self.gp_loss(interpolate_predict, interpolate_images, real_label)
            D_x = real_predict.mean().item()
            D_G_z1 = fake_predict.mean().item()
            loss_d = -real_loss + fake_loss + 10 * interpolate_loss
            loss_d.backward()
            self.dis_optimizer.step()
            #Train Generator
            self.gen_optimizer.zero_grad()
            input_noise = torch.randn(real_images.size()[0], self.noise_dim, 1, 1).to(self.device)
            fake_images = self.gen(input_noise)
            fake_predict = self.dis(fake_images)#.view(-1)
            loss_g = -fake_predict.mean()
            D_G_z2 = fake_predict.mean().item()
            loss_g.backward()
            self.gen_optimizer.step()

            G_sum_loss += loss_g.item()
            D_sum_loss += loss_d.item()
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'%\
             (batch_idx + 1, len(self.dataloader), loss_d.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))
        
        log = {
            'epoch': epoch,
            'Gen_loss': G_sum_loss,
            'Dis_loss': D_sum_loss
        }
        print("======================================================================================")
        print('FINISH EPOCH: [%d/%d] Loss_D: %.4f Loss_G: %.4f'% ((epoch + 1), self.n_epochs, D_sum_loss, G_sum_loss))
        print("======================================================================================")
        if (epoch+1)%5 == 0:
            with torch.no_grad():
                fixed_image = self.gen(self.fixed_noise)
                save_image(fixed_image.data[:25], "saved/wgan_images_%d.png" % (epoch+1), nrow = 5, normalize = True)

        return log

    def gp_loss(self, pred, img, grad_outputs):
        gradients = autograd.grad(pred, img, grad_outputs=grad_outputs, create_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')