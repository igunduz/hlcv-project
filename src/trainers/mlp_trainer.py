import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from torchvision.utils import make_grid
from .base_trainer import BaseTrainer
from utils.utils import MetricTracker

import models.mlp.model as module_arch
import models.mlp.loss as module_loss
import models.mlp.metric as module_metric

from logger import TensorboardWriter


class MLPTrainer(BaseTrainer):

    def __init__(self, config, train_loader, eval_loader=None):
        """
        Create the model, loss criterion, optimizer, and dataloaders
        And anything else that might be needed during training. (e.g. device type)
        """
        super().__init__(config)
    
        # build model architecture, then print to console
        self.model = config.init_obj('arch', module_arch)
        self.model.to(self._device)
        if len(self._device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self._device_ids)

        # Initialize the model weights based on weights_init logic
        self.model.apply(self.weights_init)

        # Simply Log the model
        self.logger.info(self.model)

        # Prepare Losses
        self.criterion = getattr(module_loss, config['loss'])

        print(self.model)

        # Prepare Optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        # Dummy scheduler thtat keeps LR constant
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) 

        # Set DataLoaders
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        
        self.log_step = 100 # arbitrary

        # setup visualization writer instance
        self.writer = None
        if config['tensorboard']:
            self.writer = TensorboardWriter(config.log_dir, self.logger)

        # Prepare Metrics
        # Basically for every metric, read the type from dict and initialize it with the given arguments
        self.metric_ftns = [getattr(module_metric, met['type'])(**met['args']) for met in config['metrics']]
        # Give all the metrics + the loss to MetricTracker.
        self.epoch_metrics = MetricTracker(keys=['loss'] + [str(m) for m in self.metric_ftns], writer=self.writer)
        self.eval_metrics = MetricTracker(keys=['loss'] + [str(m) for m in self.metric_ftns], writer=self.writer)

    def weights_init(self, m):
        """
        Initializes the model weights! Must be used with .apply of an nn.Module so that it works recursively!
        """
        if type(m) == nn.Linear: #Initialize every linear layer.
            m.weight.data.normal_(0.0, 1e-3)
            m.bias.data.fill_(0.)

    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        #######
        # Set model to train mode
        ######
        self.epoch_metrics.reset()

        self.logger.debug(f"==> Start Training Epoch {self.current_epoch}/{self.epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ")

        pbar = tqdm(total=len(self._train_loader) * self._train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (images, labels) in enumerate(self._train_loader):

            images = images.to(self._device)
            labels = labels.to(self._device)

            #################################################################################
            # TODO: Implement the training code                                             #
            # 1. Pass the images to the model                                               #
            # 2. Compute the loss using the output and the labels.                          #
            # 3. Compute gradients and update the model using the optimizer                 #
            # Use examples in https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
            #################################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.writer.set_step((self.current_epoch - 1) * len(self._train_loader) + batch_idx)
            self.epoch_metrics.update('loss', loss.item())
            for metric in self.metric_ftns:
                self.epoch_metrics.update(str(metric), metric.compute(output, labels))

            pbar.set_description(f"Train Epoch: {self.current_epoch} Loss: {loss.item():.6f}")

            if batch_idx % self.log_step == 0:
                # self.logger.debug('Train Epoch: {} Loss: {:.6f}'.format(self.current_epoch, loss.item()))
                self.writer.add_image('input', make_grid(images.cpu(), nrow=8, normalize=True))

            pbar.update(self._train_loader.batch_size)

        log_dict = self.epoch_metrics.result()
        pbar.close()
        self.lr_scheduler.step() # This doesn't to anything as we have a dummy scheduler

        self.logger.debug(f"==> Finished Epoch {self.current_epoch}/{self.epochs}.")
        
        return log_dict

    
    @torch.no_grad()
    def evaluate(self, loader=None):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluatation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        if loader is None:
            assert self._eval_loader is not None, 'loader was not given and self._eval_loader not set either!'
            loader = self._eval_loader

        self.model.eval()
        self.eval_metrics.reset()

        self.logger.debug(f"++> Evaluate at epoch {self.current_epoch} ...")

        pbar = tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, (images, labels) in enumerate(loader):
            
            images = images.to(self._device)
            labels = labels.to(self._device)

            ####################################################
            # TODO: Implement the evaluation code              #
            # 1. Pass the images to the model                  #
            # 2. Get the most confident predicted class        #
            ####################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            output = self.model(images)
            loss = self.criterion(output, labels)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.writer.set_step((self.current_epoch - 1) * len(loader) + batch_idx, 'valid')
            self.eval_metrics.update('loss', loss.item())
            for metric in self.metric_ftns:
                self.eval_metrics.update(str(metric), metric.compute(output, labels))

            pbar.set_description(f"Val Loss: {loss.item():.6f}")
            self.writer.add_image('input', make_grid(images.cpu(), nrow=8, normalize=True))

            pbar.update(loader.batch_size)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        pbar.close()
        self.logger.debug(f"++> Evaluate epoch {self.current_epoch} Finished.")
        
        return self.eval_metrics.result()

    def save_model(self, path):
        """
        Saves only the model parameters.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Saving checkpoint: {} ...".format(path))
        ###################################
        #  TODO: Load model params only
        #
        #
        ###################################
        self.logger.info("Checkpoint saved.")
        raise NotImplementedError
    
    def load_model(self, path):
        """
        Loads model params from the given path.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Loading checkpoint: {} ...".format(path))

        ###################################
        #  TODO: Load model params only
        #
        #
        ###################################

        self.logger.info("Checkpoint loaded.")
        raise NotImplementedError