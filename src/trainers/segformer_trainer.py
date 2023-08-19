import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from transformers import SegformerForSemanticSegmentation
from torchvision.utils import make_grid
from tqdm import tqdm

from .base_trainer import BaseTrainer
from utils import MetricTracker
from datasets import load_metric

class SegFormerTrainer(pl.LightningModule, BaseTrainer):

    def __init__(self, config, model, train_loader, val_loader=None):
        """
        Create the model, loss criterion, optimizer, and dataloaders
        And anything else that might be needed during training. (e.g. device type)
        """
        super(SegFormerTrainer, self).__init__()

        # build model architecture, then print to console
        # Remember to freeze a certain part of the model
        self.model = model
        # self.model = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/segformer-b0-finetuned-ade-512-512",
        #     return_dict=False,

        # )
        self.model.to(self._device)
        if len(self._device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self._device_ids)

        # Initialize the model weights based on weights_init logic
        # self.model.apply(self.weights_init)

        # Simply Log the model
        self.logger.info(self.model)

        # Prepare Losses
        self.criterion = getattr(module_loss, config['loss']) # Define loss function

        # Prepare Optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters()) # Get all trainable parameters
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params) # Define optimizer

        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer) # Define learning rate scheduler

        # Set DataLoaders from the DataLoader class
        self._train_loader = train_loader
        self._val_loader = val_loader

        self.log_step = 100 # arbitrary

        # Prepare Metrics
        # Get metrics from config file
        # Epoch Metrics are used for training
        # Eval Metrics are used for validation
        self.train_metrics = load_metric("mean_iou")
        self.val_metrics = load_metric("mean_iou")

    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        #######
        # Set model to train mode
        ######
        self.model.train()
        # self.epoch_metrics.reset()

        # Iterate over the training data
        self.logger.debug(f"==> Start Training Epoch {self.current_epoch}/{self.epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ")

        pbar = tqdm(total=len(self._train_loader) * self._train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, (images, labels) in enumerate(self._train_loader):

            images = images.to(self._device)
            labels = labels.to(self._device)

            self.optimizer.zero_grad()

            output = self.model(images)
            loss = self.criterion(output, labels)

            loss.backward()
            self.optimizer.step()

            if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(self._train_loader) + batch_idx)
            self.epoch_metrics.update('loss', loss.item())
            for metric in self.metric_ftns:
                self.epoch_metrics.update(str(metric), metric.compute(output, labels))

            pbar.set_description(f"Train Epoch: {self.current_epoch} Loss: {loss.item():.6f}")

            if batch_idx % self.log_step == 0:
                # self.logger.debug('Train Epoch: {} Loss: {:.6f}'.format(self.current_epoch, loss.item()))
                if self.writer is not None: self.writer.add_image('input_train', make_grid(images.cpu(), nrow=8, normalize=True))

            pbar.update(self._train_loader.batch_size)

        log_dict = self.epoch_metrics.result()
        pbar.close()
        self.lr_scheduler.step()

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

            output = self.model(images)
            loss = self.criterion(output, labels)

            if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(loader) + batch_idx, 'valid')
            self.eval_metrics.update('loss', loss.item())
            for metric in self.metric_ftns:
                self.eval_metrics.update(str(metric), metric.compute(output, labels))

            pbar.set_description(f"Eval Loss: {loss.item():.6f}")
            if self.writer is not None: self.writer.add_image('input_valid', make_grid(images.cpu(), nrow=8, normalize=True))

            pbar.update(loader.batch_size)

        # add histogram of model parameters to the tensorboard
        # if self.writer is not None:
        #     for name, p in self.model.named_parameters():
        #         self.writer.add_histogram(name, p, bins='auto')

        pbar.close()
        self.logger.debug(f"++> Evaluate epoch {self.current_epoch} Finished.")
        
        return self.eval_metrics.result()


