# config.py
import os
import datetime
import argparse
import json
import configparser
import utils
import re
from ast import literal_eval as make_tuple


def parse_args():
    result_path = "results/"
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = os.path.join(result_path, now)

    parser = argparse.ArgumentParser(description='Your project title goes here')

    # the following two parameters can only be provided at the command line.
    parser.add_argument('--result-path', type=str, default=result_path, metavar='', help='full path to store the results')
    parser.add_argument("-c", "--config", "--args-file", dest="config_file", default="args.txt", help="Specify a config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()

    result_path = args.result_path
    # add date and time to the result directory name
    if now not in result_path:
        result_path = os.path.join(result_path, now)

    # ======================= Data Setings =====================================
    parser.add_argument('--dataset-root-test', type=str, default=None, help='path of the data')
    parser.add_argument('--dataset-root-train', type=str, default=None, help='path of the data')
    parser.add_argument('--dataset-test', type=str, default=None, help='name of training dataset')
    parser.add_argument('--dataset-train', type=str, default=None, help='name of training dataset')
    parser.add_argument('--split_test', type=float, default=None, help='test split')
    parser.add_argument('--split_train', type=float, default=None, help='train split')
    parser.add_argument('--test-dev-percent', type=float, default=None, metavar='', help='percentage of dev in test')
    parser.add_argument('--train-dev-percent', type=float, default=None, metavar='', help='percentage of dev in train')
    parser.add_argument('--save-dir', type=str, default=os.path.join(result_path, 'Save'), metavar='', help='save the trained models here')
    parser.add_argument('--logs-dir', type=str, default=os.path.join(result_path, 'Logs'), metavar='', help='save the training log files here')
    parser.add_argument('--resume', type=str, default=None, help='full path of models to resume training')
    parser.add_argument('--nclasses', type=int, default=None, metavar='', dest='noutputs', help='number of classes for classification')
    parser.add_argument('--noutputs', type=int, default=None, metavar='', help='number of outputs, i.e. number of classes for classification')
    parser.add_argument('--input-filename-test', type=str, default=None, help='input test filename for filelist and folderlist')
    parser.add_argument('--label-filename-test', type=str, default=None, help='label test filename for filelist and folderlist')
    parser.add_argument('--input-filename-train', type=str, default=None, help='input train filename for filelist and folderlist')
    parser.add_argument('--label-filename-train', type=str, default=None, help='label train filename for filelist and folderlist')
    parser.add_argument('--loader-input', type=str, default=None, help='input loader')
    parser.add_argument('--loader-label', type=str, default=None, help='label loader')
    parser.add_argument('--dataset-options', type=json.loads, default=None, metavar='', help='additional model-specific parameters, i.e. \'{"gauss": 1}\'')

    # ======================= Network Model Setings ============================
    parser.add_argument('--model-type', type=str, default=None, help='type of network')
    parser.add_argument('--model-options', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--loss-type', type=str, default=None, help='loss method')
    parser.add_argument('--loss-options', type=json.loads, default={}, metavar='', help='loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--evaluation-type', type=str, default=None, help='evaluation method')
    parser.add_argument('--evaluation-options', type=json.loads, default={}, metavar='', help='evaluation-specific parameters, i.e. \'{"topk": 1}\'')
    parser.add_argument('--resolution-high', type=int, default=None, help='image resolution height')
    parser.add_argument('--resolution-wide', type=int, default=None, help='image resolution width')
    parser.add_argument('--ndim', type=int, default=None, help='number of feature dimensions')
    parser.add_argument('--nunits', type=int, default=None, help='number of units in hidden layers')
    parser.add_argument('--dropout', type=float, default=None, help='dropout parameter')
    parser.add_argument('--length-scale', type=float, default=None, help='length scale')
    parser.add_argument('--tau', type=float, default=None, help='Tau')

    # ======================= Training Settings ================================
    parser.add_argument('--cuda', type=utils.str2bool, default=None, help='run on gpu')
    parser.add_argument('--ngpu', type=int, default=None, help='number of gpus to use')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size for training')
    parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--niters', type=int, default=None, help='number of iterations at test time')
    parser.add_argument('--epoch-number', type=int, default=None, help='epoch number')
    parser.add_argument('--nthreads', type=int, default=None, help='number of threads for data loading')
    parser.add_argument('--manual-seed', type=int, default=None, help='manual seed for randomness')

    # ===================== Visualization Settings =============================
    parser.add_argument('-p', '--port', type=int, default=None, metavar='', help='port for visualizing training at http://localhost:port')
    parser.add_argument('--env', type=str, default='', metavar='', help='environment for visualizing training at http://localhost:port')

    # ======================= Hyperparameter Setings ===========================
    parser.add_argument('--learning-rate', type=float, default=None, help='learning rate')
    parser.add_argument('--optim-method', type=str, default=None, help='the optimization routine ')
    parser.add_argument('--optim-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters, i.e. \'{"lr": 0.001}\'')
    parser.add_argument('--scheduler-method', type=str, default=None, help='cosine, step, exponential, plateau')
    parser.add_argument('--scheduler-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters')

    # ======================== Main Setings ====================================
    parser.add_argument('--log-type', type=str, default='traditional', metavar='', help='allows to select logger type, traditional or progressbar')
    parser.add_argument('--same-env', type=utils.str2bool, default='No', metavar='', help='does not add date and time to the visdom environment name')
    parser.add_argument('-s', '--save', '--save-results', type=utils.str2bool, dest="save_results", default='No', metavar='', help='save the arguments and the results')

    if os.path.exists(args.config_file):
        config = configparser.ConfigParser()
        config.read([args.config_file])
        defaults = dict(config.items("Arguments"))
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)

    # add date and time to the name of Visdom environment and the result
    if args.env is '':
        args.env = args.model_type
    if not args.same_env:
        args.env += '_' + now
    args.result_path = result_path

    # refine tuple arguments: this section converts tuples that are
    #                         passed as string back to actual tuples.
    pattern = re.compile('^\(.+\)')

    for arg_name in vars(args):
        # print(arg, getattr(args, arg))
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str) and pattern.match(arg_value):
            setattr(args, arg_name, make_tuple(arg_value))
            print(arg_name, arg_value)
        elif isinstance(arg_value, dict):
            dict_changed = False
            for key, value in arg_value.items():
                if isinstance(value, str) and pattern.match(value):
                    dict_changed = True
                    arg_value[key] = make_tuple(value)
            if dict_changed:
                setattr(args, arg_name, arg_value)

    return args
