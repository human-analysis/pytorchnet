# test.py

import time
import torch
from torch.autograd import Variable
import plugins


class Tester:
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation
        self.save_results = args.save_results

        self.env = args.env
        self.port = args.port
        self.dir_save = args.save_dir
        self.log_type = args.log_type

        self.cuda = args.cuda
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size

        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

        # for classification
        self.labels = torch.zeros(self.batch_size).long()
        self.inputs = torch.zeros(
            self.batch_size,
            self.resolution_high,
            self.resolution_wide
        )

        if args.cuda:
            self.labels = self.labels.cuda()
            self.inputs = self.inputs.cuda()

        self.inputs = Variable(self.inputs, volatile=True)
        self.labels = Variable(self.labels, volatile=True)

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger.txt',
            self.save_results
        )
        self.params_loss = ['Loss', 'Accuracy']
        self.log_loss.register(self.params_loss)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Loss': {'dtype': 'running_mean'},
            'Accuracy': {'dtype': 'running_mean'}
        }
        self.monitor.register(self.params_monitor)

        # visualize testing
        self.visualizer = plugins.Visualizer(self.port, self.env, 'Test')
        self.params_visualizer = {
            'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss',
                     'layout': {'windows': ['train', 'test'], 'id': 1}},
            'Accuracy': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'accuracy',
                         'layout': {'windows': ['train', 'test'], 'id': 1}},
            'Test_Image': {'dtype': 'image', 'vtype': 'image',
                           'win': 'test_image'},
            'Test_Images': {'dtype': 'images', 'vtype': 'images',
                            'win': 'test_images'},
        }
        self.visualizer.register(self.params_visualizer)

        if self.log_type == 'traditional':
            # display training progress
            self.print_formatter = 'Test [%d/%d][%d/%d] '
            for item in self.params_loss:
                self.print_formatter += item + " %.4f "
        elif self.log_type == 'progressbar':
            # progress bar message formatter
            self.print_formatter = '({}/{})' \
                                   ' Load: {:.6f}s' \
                                   ' | Process: {:.3f}s' \
                                   ' | Total: {:}' \
                                   ' | ETA: {:}'
            for item in self.params_loss:
                self.print_formatter += ' | ' + item + ' {:.4f}'

        self.evalmodules = []
        self.losses = {}

    def model_eval(self):
        self.model.eval()

    def test(self, epoch, dataloader):
        dataloader = dataloader['test']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()

        if self.log_type == 'progressbar':
            # progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<10}'.format('Test'), max=len(dataloader))
        end = time.time()

        for i, (inputs, labels) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end

            ############################
            # Evaluate Network
            ############################

            batch_size = inputs.size(0)
            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(labels.size()).copy_(labels)

            self.model.zero_grad()
            output = self.model(self.inputs)
            loss = self.criterion(output, self.labels)

            acc = self.evaluation(output, self.labels)

            self.losses['Accuracy'] = acc
            self.losses['Loss'] = loss.data[0]
            self.monitor.update(self.losses, batch_size)

            if self.log_type == 'traditional':
                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i, len(dataloader)] +
                    [self.losses[key] for key in self.params_monitor]))
            elif self.log_type == 'progressbar':
                # update progress bar
                batch_time = time.time() - end
                processed_data_len += len(inputs)
                bar.suffix = self.print_formatter.format(
                    *[processed_data_len, len(dataloader.sampler), data_time,
                      batch_time, bar.elapsed_td, bar.eta_td] +
                     [self.losses[key] for key in self.params_monitor]
                )
                bar.next()
                end = time.time()

        if self.log_type == 'progressbar':
            bar.finish()

        loss = self.monitor.getvalues()
        self.log_loss.update(loss)
        loss['Test_Image'] = inputs[0]
        loss['Test_Images'] = inputs
        self.visualizer.update(loss)
        return self.monitor.getvalues('Loss')
