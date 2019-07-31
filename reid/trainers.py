from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterions):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterions = criterions

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets, epoch)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            #add gradient clip for lstm
            for param in self.model.parameters():
                try:
                    param.grad.data.clamp(-1., 1.)
                except:
                    continue

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(*inputs) #outputs=[x1,x2,x3]
        #new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)
        # x2 global feature cross entropy loss
        loss_global = self.criterions[1](outputs[1], targets,epoch)
        #prec_global, = accuracy(outputs[1].data, targets.data)
        #prec_global = prec_global[0]
        # x3 local lstm feature cross entropy loss
        loss_local = self.criterions[2](outputs[2], targets)

        return loss_tri+loss_global+0.2*loss_local, prec_tri
        

        #new added by wc

        '''
        if isinstance(self.criterions, torch.nn.CrossEntropyLoss):
            loss = self.criterions(outputs[1], targets)
            prec, = accuracy(outputs[1].data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterions, OIMLoss):
            loss, outputs = self.criterions(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterions, TripletLoss):
            loss, prec = self.criterions(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
        '''
