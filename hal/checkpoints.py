# checkpoints.py

import os
import torch
import operator


class Checkpoints:
    def __init__(self, args):
        self.save_dir = args.save_dir
        self.model_filename = args.resume
        self.save_results = args.save_results
        self.max_history = args.checkpoint_max_history

        self.best_epoch = None
        self.best_metric = None
        self.checkpoint_files = []
        
        self.extension = '.pth.tar'
        self.save_prefix = 'checkpoint'
        self.decreasing = args.monitor['decreasing']
        self.cmp = operator.lt if self.decreasing else operator.gt

        if self.save_results and not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def latest(self, name):
        if name == 'resume':
            return self.model_filename

    def save(self, epoch, model, metric=None):
        last_save_path = os.path.join(self.save_dir, 'last' + self.extension)
        save_state = {
            'epoch': epoch,
            # 'state_dict': [model[key].state_dict() for key in model],
            'state_dict': model,
            'metric': metric
        }
        torch.save(save_state, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.save_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            print(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(
                    self.save_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def load(self, model, filename, strict=True):
        if os.path.isfile(filename):
            resume_epoch = None
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=strict)
                if 'epoch' in checkpoint:
                    resume_epoch = checkpoint['epoch'] + 1
            else:
                model.load_state_dict(checkpoint, strict=strict)
            return resume_epoch
        raise (Exception("=> no checkpoint found at '{}'".format(filename)))

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                print("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                print("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]
