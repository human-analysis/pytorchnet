# utils.py

import os
import csv
import argparse
from inspect import getframeinfo, stack
import json
import sys
import psutil
import signal


def setup_graceful_exit():
    # handle Ctrl-C signal
    signal.signal(signal.SIGINT, ctrl_c_handler)


def cleanup():
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            os.kill(int(child.pid), signal.SIGKILL)
        except OSError as ex:
            raise Exception("wasn't able to kill the child process (pid:{}).".format(child.pid))
    #     # os.waitpid(child.pid, os.P_ALL)
    print('\x1b[?25h', end='', flush=True)  # show cursor
    sys.exit(0)


def ctrl_c_handler(*kargs):
    # try to gracefully terminate the program
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    cleanup()


def isnan(x):
    return x != x


def _debuginfo(self, *message):
    """Prints the current filename and line number in addition to debugging
    messages."""
    caller = getframeinfo(stack()[1][0])
    print('\033[92m', caller.filename, '\033[0m', caller.lineno,
          '\033[95m', self.__class__.__name__, '\033[94m', message, '\033[0m')


def readcsvfile(filename, delimiter=','):
    with open(filename, 'r') as f:
        content = []
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            content.append(row)
    f.close()
    return content


def readtextfile(filename):
    with open(filename) as f:
        content = f.readlines()
    f.close()
    return content


def writetextfile(data, filename, path=None):
    """If path is provided, it will make sure the path exists before writing
    the file."""
    if path:
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, filename)
    with open(filename, 'w') as f:
        f.writelines(data)
    f.close()


def delete_file(filename):
    if os.path.isfile(filename) is True:
        os.remove(filename)


def eformat(f, prec, exp_digits):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def saveargs(args: object) -> object:
    path = args.logs_dir
    varargs = '[Arguments]\n\n'
    # TODO: organize the parameters into groups
    for par in vars(args):
        if getattr(args, par) is None or par in ['save_dir', 'logs_dir',
                                                 'save_results', 'result_path',
                                                 'config_file']:
            continue
        elif par in ('model_options', 'loss_options', 'evaluation_options',
                     'dataset_options'):
            varargs += '%s = %s\n' % (par, json.dumps(getattr(args, par)))
        else:
            varargs += '%s = %s\n' % (par, getattr(args, par))
    writetextfile(varargs, 'args.txt', path)


def file_exists(filename):
    return os.path.isfile(filename)


def str2bool(v):
    """A Parser for boolean values with argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plotlify(fig, env='main', win='mywin'):
    fig = {key: fig[key] for key in fig.keys()}
    fig['win'] = win
    fig['eid'] = env

    return fig

def print_model_parameters(model):
    len1 = 0;len2 = 0
    values = []
    for key in model:
        values.append(key)
        values.append(sum(p.numel() for p in model[key].parameters()) / 1000000.0)

    len_dash = 31

    print_format = '| {:<10} | {:<14} | \n'
    print_format += "-"*len_dash + '\n'

    for key in model:
        print_format += '| {:<10} | {:<14.4f} |'
    
    print("-"*len_dash)
    print(print_format.format(
        *['Model Name', 'Parameters (M)'] + 
        values
    ))
    print("-"*len_dash)