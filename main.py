#!/usr/bin/env python
# main.py
"""
Training/Testing Processor

This module provides a comprehensive training and testing framework for skeleton-based
action recognition using Spatial Temporal Graph Convolutional Networks. It supports:
- Model training with configurable optimizers and learning rate scheduling
- Model evaluation with top-k accuracy metrics
- TensorBoard logging and visualization
- Cross-validation on multiple datasets (NTU RGB+D, Kinetics, etc.)
- Model checkpointing and best model selection
- Comprehensive metrics including confusion matrices and per-class accuracy
"""

from __future__ import print_function
from typing import Dict, List, Optional, Union, Tuple, Any
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# PyTorch imports
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)



def init_seed(seed: int) -> None:
    """
    Initialize random seeds for reproducible results across all random number generators.
    
    Args:
        seed: Random seed value to use across all libraries
        
    Note:
        Sets deterministic behavior for CUDA operations which may impact performance
        but ensures reproducible results across runs.
    """
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str: str) -> type:
    """
    Dynamically import a class from a module string.
    
    Args:
        import_str: Module path in format 'module.submodule.ClassName'
        
    Returns:
        type: The imported class object
        
    Raises:
        ImportError: If the class cannot be found in the specified module
        
    Example:
        Model = import_class('net.st_gcn.Model')
    """
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v: Union[str, bool]) -> bool:
    """
    Convert string representation to boolean value.
    
    Args:
        v: String or boolean value to convert
        
    Returns:
        bool: Converted boolean value
        
    Raises:
        argparse.ArgumentTypeError: If the input string cannot be converted to boolean
        
    Example:
        str2bool('true')  # Returns True
        str2bool('no')    # Returns False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured parser with all training/testing options
        
    Note:
        Parameter priority: command line > config file > default values
        Configuration covers model settings, training hyperparameters, data loading,
        and evaluation options.
    """
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    
    # Workspace and configuration
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-subject/default.yaml',
        help='path to the configuration file')

    # Training/testing phase
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # Visualization and debugging
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # Data feeder configuration
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # Model configuration
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # Optimization parameters
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser


class Processor:
    """
    Main processor class for Skeleton-based Action Recognition.
    
    This class handles the complete training and testing pipeline including:
    - Model initialization and weight loading
    - Data loading and preprocessing
    - Training loop with loss computation and backpropagation  
    - Evaluation with multiple metrics
    - Model checkpointing and best model selection
    - TensorBoard logging and visualization
    
    Attributes:
        arg: Parsed command line arguments containing all configuration
        model: The neural network model (ST-GCN or variants)
        optimizer: Optimization algorithm (SGD/Adam)
        loss: Loss function (CrossEntropyLoss)
        data_loader: Dictionary containing train/test data loaders
        best_acc: Best validation accuracy achieved during training
        best_acc_epoch: Epoch number where best accuracy was achieved
        global_step: Global training step counter for logging
    """

    def __init__(self, arg: argparse.Namespace) -> None:
        """
        Initialize the processor with configuration arguments.
        
        Args:
            arg: Parsed command line arguments containing all settings
        """
        self.arg = arg
        self.save_arg()
        
        # Setup TensorBoard logging
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        
        self.global_step: int = 0
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
            
        self.lr: float = self.arg.base_lr
        self.best_acc: float = 0
        self.best_acc_epoch: int = 0

        # Move model to GPU and setup multi-GPU if available
        self.model = self.model.cuda(self.output_device)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self) -> None:
        """
        Load training and testing datasets using the specified feeder.
        
        Creates PyTorch DataLoaders for both training and testing phases
        with appropriate batch sizes, shuffling, and worker processes.
        """
        Feeder = import_class(self.arg.feeder)
        self.data_loader: Dict[str, torch.utils.data.DataLoader] = dict()
        
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
                
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self) -> None:
        """
        Load and initialize the model architecture and loss function.
        
        Handles:
        - Dynamic model class import and initialization
        - Pre-trained weight loading with optional weight filtering
        - Multi-GPU device configuration
        - Cross-entropy loss function setup
        """
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        
        # Import and initialize model
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        # Load pre-trained weights if specified
        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            # Remove 'module.' prefix from multi-GPU saved models
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            # Remove specified weights from loading
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Successfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            # Load weights with error handling
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self) -> None:
        """
        Initialize the optimizer based on configuration.
        
        Supports:
        - SGD with momentum and Nesterov acceleration
        - Adam optimizer
        - Weight decay regularization
        - Warm-up learning rate scheduling
        """
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {self.arg.optimizer}')

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self) -> None:
        """
        Save configuration arguments to YAML file for reproducibility.
        
        Creates the work directory if it doesn't exist and saves all
        command line arguments and config file settings.
        """
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch: int) -> float:
        """
        Adjust learning rate based on epoch and warm-up schedule.
        
        Args:
            epoch: Current training epoch number
            
        Returns:
            float: Updated learning rate value
            
        Learning rate schedule:
        - Warm-up phase: Linear increase from 0 to base_lr
        - Normal phase: Step decay at specified epochs
        """
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError(f'Unsupported optimizer: {self.arg.optimizer}')

    def print_time(self) -> None:
        """Print current local time to log."""
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str_msg: str, print_time: bool = True) -> None:
        """
        Print message to console and optionally save to log file.
        
        Args:
            str_msg: Message string to print/log
            print_time: Whether to prepend timestamp to message
        """
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str_msg = "[ " + localtime + ' ] ' + str_msg
        print(str_msg)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str_msg, file=f)

    def record_time(self) -> float:
        """
        Record current time for timing measurements.
        
        Returns:
            float: Current timestamp
        """
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self) -> float:
        """
        Calculate elapsed time since last record_time() call.
        
        Returns:
            float: Elapsed time in seconds
        """
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch: int, save_model: bool = False) -> None:
        """
        Execute one training epoch.
        
        Args:
            epoch: Current epoch number
            save_model: Whether to save model weights after this epoch
            
        Performs:
        - Forward pass through all training batches
        - Loss computation and backpropagation
        - Accuracy calculation and logging
        - Learning rate adjustment
        - Optional model checkpointing
        """
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value: List[float] = []
        acc_value: List[float] = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # Forward pass
            output = self.model(data)
            loss = self.loss(output, label)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            # Calculate accuracy
            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # Log learning rate
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # Print epoch statistics
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(
                np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # Save model checkpoint
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch: int, save_score: bool = False, loader_name: List[str] = ['test'], 
             wrong_file: Optional[str] = None, result_file: Optional[str] = None) -> None:
        """
        Evaluate model performance on specified datasets.
        
        Args:
            epoch: Current epoch number for logging
            save_score: Whether to save prediction scores to file
            loader_name: List of data loader names to evaluate on
            wrong_file: Optional file path to save incorrect predictions
            result_file: Optional file path to save all predictions
            
        Computes:
        - Loss and top-k accuracy metrics
        - Per-class accuracy and confusion matrix
        - Updates best accuracy tracking
        - Logs results to TensorBoard and files
        """
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
            
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        
        for ln in loader_name:
            loss_value: List[float] = []
            score_frag: List[np.ndarray] = []
            label_list: List[np.ndarray] = []
            pred_list: List[np.ndarray] = []
            step: int = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                # Save predictions to files if specified
                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
                            
            # Compute final metrics
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            
            # Update best accuracy
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            # Save scores
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            
            # Print top-k accuracies
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            # Save prediction scores
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # Compute per-class accuracy and confusion matrix
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self) -> None:
        """
        Start the training or testing process based on the specified phase.
        
        Training phase:
        - Runs training and evaluation loops for specified epochs
        - Saves model checkpoints and tracks best performance
        - Evaluates final best model and saves detailed results
        
        Testing phase:
        - Loads specified weights and evaluates on test set
        - Saves prediction results and error analysis
        """
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            
            def count_parameters(model: nn.Module) -> int:
                """Count trainable parameters in the model."""
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            
            # Training loop
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # Evaluate best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            # Print final results
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def main() -> None:
    """
    Main function to parse arguments and start the training/testing process.
    
    Workflow:
    1. Parse command line arguments
    2. Load configuration from YAML file if specified
    3. Initialize random seed for reproducibility
    4. Create processor instance and start training/testing
    """
    parser = get_parser()

    # Load arguments from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()