import torch
import os

from abc import abstractmethod
from numpy import inf
from utils.utils import prepare_device


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config):
        
        self.config = config
        self.logger = self.config.get_logger('trainer', config['trainer']['verbosity'])

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.eval_period = cfg_trainer['eval_period']

        self.checkpoint_dir = config.save_dir

        self.start_epoch = 1
        self.current_epoch = 1

        # prepare for (multi-device) GPU training
        # This part doesn't do anything if you don't have a GPU.
        self._device, self._device_ids = prepare_device(config['n_gpu'])

    @abstractmethod
    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            result = self._train_epoch()

            # save logged informations into log dict
            log = {'epoch': self.current_epoch}
            log.update(result)

            if self.do_evaluate():
                result = self.evaluate()
                # save eval information to the log dict as well
                log.update({f'eval_{key}': value for key, value in result.items()})    

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if self.current_epoch % self.save_period == 0:
                path = os.path.join(self.checkpoint_dir, f'E{self.current_epoch}_model.ckpt')
                self.save_model(path=path)

    def do_evaluate(self):
        """
        Based on the self.current_epoch and self.eval_interval, determine if we should evaluate.
        You can take hint from saving logic implemented in BaseTrainer.train() method

        returns a Boolean
        """
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, loader=None):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        raise NotImplementedError

    def save_model(self, path):
        """
        Saves only the model parameters.
        : param path: path to save model (including filename.)
        """
        raise NotImplementedError
    
    def load_model(self, path):
        """
        Loads model params from the given path.
        : param path: path to save model (including filename.)
        """
        raise NotImplementedError

    def _save_checkpoint(self, path='checkpoints/ckpt.pth'):
        """
        Saving TRAINING checkpoint. Including the model params and other training stats 
        (optimizer, current epoch, etc.)

        :param path: if True, rename the saved checkpoint to 'model_best.pth'
        """
        raise RuntimeError("Not for this assignment!")

    def _resume_checkpoint(self, path='checkpoints/ckpt.pth'):
        """
        Loads TRAINING checkpoint. Including the model params and other training stats 
        (optimizer, current epoch, etc.)

        :param path: Checkpoint path to be resumed
        """
        raise RuntimeError("Not for this assignment!")
