# train.py

import time
import torch
import torch.optim as optim
from torch.autograd import Variable
import plugins


class Trainer:
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.save_results = args.save_results
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation

        self.port = args.port
        self.env = args.env
        self.dir_save = args.save_dir

        self.cuda = args.cuda
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size

        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

        self.lr = args.learning_rate
        self.momentum = args.momentum
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.weight_decay = args.weight_decay
        self.optim_method = args.optim_method

        if self.optim_method == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        elif self.optim_method == 'RMSprop':
            self.optimizer = optim.RMSprop(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
        elif self.optim_method == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                momentum=self.momentum, nesterov=True
            )
        else:
            raise (Exception("Unknown Optimization Method"))

        # for classification
        self.labels = torch.zeros(self.batch_size).long()
        self.inputs = torch.zeros(self.batch_size, self.resolution_high, self.resolution_wide)

        if args.cuda:
            self.labels = self.labels.cuda()
            self.inputs = self.inputs.cuda()

        self.inputs = Variable(self.inputs)
        self.labels = Variable(self.labels)

        # logging training
        self.log_loss = plugins.Logger(args.logs_dir, 'TrainLogger.txt', self.save_results)
        self.params_loss = ['Loss', 'Accuracy']
        self.log_loss.register(self.params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Loss': {'dtype': 'running_mean'},
            'Accuracy': {'dtype': 'running_mean'}
        }
        self.monitor.register(self.params_monitor)

        # visualize training
        self.visualizer = plugins.Visualizer(self.port, self.env, 'Train')
        self.params_visualizer = {
            'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss', 'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Accuracy': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'accuracy', 'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Train_Image': {'dtype': 'image', 'vtype': 'image', 'win': 'train_image'},
            'Train_Images': {'dtype': 'images', 'vtype': 'images', 'win': 'train_images'},
        }
        self.visualizer.register(self.params_visualizer)

        # progress bar message formatter
        self.print_formatter = '({}/{})' \
                               ' Load: {:.6f}s' \
                               ' | Process: {:.3f}s' \
                               ' | Total: {:}' \
                               ' | ETA: {:}'
        for item in self.params_loss:
            self.print_formatter += ' | ' + item + ' {:.4f}'
        self.print_formatter += ' | lr: {:.2e}'

        self.evalmodules = []
        self.losses = {}

    def learning_rate(self, epoch):
        # training schedule
        return self.lr * (
            (0.1 ** int(epoch >= 60)) *
            (0.1 ** int(epoch >= 120)) *
            (0.1 ** int(epoch >= 160))
        )

    def get_optimizer(self, epoch, optimizer):
        lr = self.learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def model_train(self):
        self.model.train()

    def train(self, epoch, dataloader):
        dataloader = dataloader['train']
        self.monitor.reset()
        self.optimizer = self.get_optimizer(epoch + 1, self.optimizer)

        # switch to train mode
        self.model_train()

        # Progress bar
        processed_data_len = 0
        bar = plugins.Bar('{:<10}'.format('Train'), max=len(dataloader))
        end = time.time()

        for i, (inputs, labels) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end

            ############################
            # Update network
            ############################

            batch_size = inputs.size(0)
            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(labels.size()).copy_(labels)

            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step(loss.data[0])

            acc = self.evaluation(outputs, self.labels)

            self.losses['Accuracy'] = acc
            self.losses['Loss'] = loss.data[0]
            self.monitor.update(self.losses, batch_size)

            # update progress bar
            batch_time = time.time() - end
            processed_data_len += len(inputs)

            bar.suffix = self.print_formatter.format(
                *[processed_data_len, len(dataloader.sampler), data_time,
                  batch_time, bar.elapsed_td, bar.eta_td] +
                 [self.losses[key] for key in self.params_monitor] +
                 [self.optimizer.param_groups[-1]['lr']]
            )
            bar.next()
            end = time.time()

        bar.finish()

        loss = self.monitor.getvalues()
        self.log_loss.update(loss)
        loss['Train_Image'] = inputs[0]
        loss['Train_Images'] = inputs
        self.visualizer.update(loss)
        return self.monitor.getvalues('Loss')
