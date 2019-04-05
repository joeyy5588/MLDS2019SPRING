import torch
import logging
import os
import random
import torch.nn as nn
from torchvision.utils import save_image
from models import BLEU_metric
from utils import ensure_dir

class Trainer:    
    def __init__(self, model, index_to_word, dataloader, val_dataloader, opt):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.n_epochs = opt.n_epochs
        self.device = self._prapare_gpu()
        self.model = model
        self.i2w = index_to_word
        self.optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
        self.criterion = nn.NLLLoss(ignore_index = 0) # ignore padding
        self.metrics = BLEU_metric
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.resume)

    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.model.to(self.device)
        ensure_dir(opt.save_dir)
        self.logger.info('[STRUCTURE]')
        self.logger.info(self.model)
        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log, sen = self._train_epoch(i)
            val_log, val_sen = self._val_epoch(i)
            merged_log = {**log, **val_log}
            self.logger.info('======================================================================================')
            self.logger.info('[STATISTIC]')
            self._log_dict(merged_log)
            self.logger.info('[RESULT]')
            self.logger.info('Train')
            self._log_random_sentences(sen, num = 5)
            self.logger.info('Validation')
            self._log_random_sentences(val_sen, num = 5)
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

    def _log_dict(self, d):
        for (k, v) in d.items():
            self.logger.info('\t{}: {}'.format(k, v))

    def _log_random_sentences(self, sen, num = 5):
        n_sen = sen['gt'].shape[0]
        rand_idxs = random.sample(range(n_sen), num)
        for n, i in enumerate(rand_idxs):
            self.logger.info('\tSample {}:'.format(n))
            self.logger.info('\tGround Truth: {}'.format(self._idxs_to_sentence(sen['gt'][i])))
            self.logger.info('\tPrediction: {}'.format(self._idxs_to_sentence(sen['pred'][i])))

    def _schedule_sampling(self, epoch):
        #p = 1 * (1000 - epoch) / 1000 # Linear
        p = 1
        return p

    def _train_epoch(self, epoch):
        device, model, criterion, optimizer = self.device, self.model, self.criterion, self.optimizer
        model.train()
        dataloader = self.dataloader
        total_loss, total_score = 0, 0
        total_sen = {'gt': torch.tensor([], dtype = torch.long).to(device), 'pred': torch.tensor([], dtype = torch.long).to(device)}
        for i, ((feat, idxs), sens) in enumerate(dataloader):
            optimizer.zero_grad()
            feat, idxs = feat.to(device), idxs.to(device)
            p = self._schedule_sampling(epoch)
            out = model(feat, idxs, p)[:len(sens), :, :]
            total_sen['gt'] = torch.cat((total_sen['gt'], idxs), dim = 0)
            total_sen['pred'] = torch.cat((total_sen['pred'], torch.argmax(out, dim = 2)), dim = 0)
            score = self.metrics(self._batch_to_sentences(torch.argmax(out, dim = 2)), sens) 
            total_score += score
            # use 2d nll_loss
            loss = criterion(out.transpose(1, 2).unsqueeze(3), idxs.unsqueeze(2))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # self.logger.info('[EPOCH {}/{}] [BATCH: {}/{}] [LOSS: {:.3f}] [SCORE: {:.3F}]'.format(epoch, self.n_epochs, i, len(self.dataloader), loss.item(), score))
        
        log = {
            'epoch': epoch,
            'loss': total_loss / len(dataloader),
            'score': total_score / len(dataloader)
        }

        return log, total_sen

    def _val_epoch(self, epoch):
        device, model, criterion = self.device, self.model, self.criterion
        model.eval()
        dataloader = self.val_dataloader
        total_loss, total_score = 0, 0
        total_sen = {'gt': torch.tensor([], dtype = torch.long).to(device), 'pred': torch.tensor([], dtype = torch.long).to(device)}
        for i, ((feat, idxs), sens) in enumerate(dataloader):
            feat, idxs = feat.to(device), idxs.to(device)
            out = model(feat, idxs)[:len(sens), :, :]
            total_sen['gt'] = torch.cat((total_sen['gt'], idxs), dim = 0)
            total_sen['pred'] = torch.cat((total_sen['pred'], torch.argmax(out, dim = 2)), dim = 0)
            score = self.metrics(self._batch_to_sentences(torch.argmax(out, dim = 2)), sens) 
            total_score += score
            # use 2d nll_loss
            loss = criterion(out.transpose(1, 2).unsqueeze(3), idxs.unsqueeze(2))
            total_loss += loss.item()
            # self.logger.info('[EPOCH {}/{}] [BATCH: {}/{}] [LOSS: {:.3f}] [SCORE: {:.3F}]'.format(epoch, self.n_epochs, i, len(self.dataloader), loss.item(), score))
        
        log = {
            'epoch': epoch,
            'val_loss': total_loss / len(dataloader),
            'val_score': total_score / len(dataloader)
        }

        return log, total_sen
    
    def _batch_to_sentences(self, batch):
        sen_list = []
        for i in range(batch.shape[0]):
            sen_list.append(self._idxs_to_sentence(batch[i]))
        return sen_list
    
    def _idxs_to_sentence(self, sen_idxs):
        i2w = self.i2w
        sen = []
        for i in sen_idxs:
            w = i2w[i.item()]
            if w == '<EOS>': break
            sen.append(w)

        return ' '.join(sen)

    def _prapare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.begin_epoch = checkpoint['log'][-1]['epoch']
            self.all_log = checkpoint['log']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')