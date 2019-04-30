import torch
import logging
import os
import random
import torch.nn as nn
import math

class Trainer:    
    def __init__(self, model, lang, dataloader, opt):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.dataloader = dataloader
        self.n_epochs = opt.n_epochs
        self.device = self._prepare_gpu()
        self.model = model
        self.i2w = lang.i2w
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = opt.lr)
        self.criterion = nn.NLLLoss(ignore_index = lang.w2i['<PAD>']) # ignore padding
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.checkpoint)

    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.model.to(self.device)
        self.logger.info('[STRUCTURE]')
        self.logger.info(self.model)

        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log, sen = self._train_epoch(i)
            merged_log = {**log}
            self.logger.info('======================================================================================')
            self.logger.info('[STATISTIC]')
            self._log_dict(merged_log)
            self.logger.info('[RESULT]')
            self.logger.info('Train')
            self._log_random_sentences(sen, num = 5)
            self.logger.info('[SAVED]')
            all_log.append(merged_log)
            checkpoint = {
                'log': all_log,
                'state_dict': self.model.state_dict(),
            }
            check_path = os.path.join(opt.save_dir, 'checkpoint.pth')
            torch.save(checkpoint, check_path)
            self._log_dict({'checkpoint': check_path})
            self.logger.info('======================================================================================\n')

    def _train_epoch(self, epoch):
        device, model, criterion, optimizer = self.device, self.model, self.criterion, self.optimizer
        dataloader = self.dataloader
        total_loss = 0
        total_sen = {
            'question': torch.tensor([], dtype = torch.long).to(device),
            'gt': torch.tensor([], dtype = torch.long).to(device),
            'pred': torch.tensor([], dtype = torch.long).to(device)
        }
        
        model.train()
        for i, ((q_idxs, a_idxs)) in enumerate(dataloader):
            optimizer.zero_grad()
            q_idxs, a_idxs = q_idxs.to(device), a_idxs.to(device)
            out = model(q_idxs, a_idxs)

            total_sen['question'] = torch.cat((total_sen['question'], q_idxs), dim = 0)
            total_sen['gt'] = torch.cat((total_sen['gt'], a_idxs), dim = 0)
            total_sen['pred'] = torch.cat((total_sen['pred'], torch.argmax(out, dim = 2)), dim = 0)

            # use 2d nll_loss 
            loss = criterion(out.transpose(1, 2).unsqueeze(3), a_idxs.unsqueeze(2))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            self.logger.info('[EPOCH {}/{}] [BATCH: {}/{}] [LOSS: {:.3f}]'.format(epoch, self.begin_epoch + self.n_epochs, i, len(self.dataloader), loss.item()))
        log = {
            'epoch': epoch,
            'loss': total_loss / len(dataloader),
        }
        return log, total_sen

    def _log_random_sentences(self, sen, num = 5):
        n_sen = sen['gt'].shape[0]
        rand_idxs = random.sample(range(n_sen), num)
        for n, i in enumerate(rand_idxs):
            self.logger.info('\tSample {}:'.format(n))
            self.logger.info('\tQuestion: {}'.format(self._idxs_to_sentence(sen['question'][i])))
            self.logger.info('\tGround Truth: {}'.format(self._idxs_to_sentence(sen['gt'][i])))
            self.logger.info('\tPrediction: {}'.format(self._idxs_to_sentence(sen['pred'][i])))
    
    def _idxs_to_sentence(self, sen_idxs):
        i2w = self.i2w
        sen = []
        for i in sen_idxs:
            w = i2w[i.item()]
            if w == '<EOS>': break
            sen.append(w)
        return ' '.join(sen)

    def _log_dict(self, d):
        for (k, v) in d.items():
            self.logger.info('\t{}: {}'.format(k, v))

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