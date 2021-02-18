# visualizer.py

from torch.utils.tensorboard import SummaryWriter

__all__ = ['VisualizerTensorboard']


class VisualizerTensorboard:
    def __init__(self, opts):
        self.dtype = {}
        self.iteration = 1
        self.writer = SummaryWriter(opts.logs_dir)

    def register(self, modules):
        # here modules are assumed to be a dictionary
        for key in modules:
            self.dtype[key] = modules[key]['dtype']

    def update(self, modules):
        for key, value in modules:
            if self.dtype[key] == 'scalar':
                self.writer.add_scalar(key, value, self.iteration)
            elif self.dtype[key] == 'scalars':
                self.writer.add_scalars(key, value, self.iteration)
            elif self.dtype[key] == 'histogram':
                self.writer.add_histogram(key, value, self.iteration)
            elif self.dtype[key] == 'image':
                self.writer.add_image(key, value, self.iteration)
            elif self.dtype[key] == 'images':
                self.writer.add_images(key, value, self.iteration)
            elif self.dtype[key] == 'figure':
                self.writer.add_figure(key, value, self.iteration)
            elif self.dtype[key] == 'video':
                self.writer.add_video(key, value, self.iteration)
            elif self.dtype[key] == 'audio':
                self.writer.add_audio(key, value, self.iteration)
            elif self.dtype[key] == 'text':
                self.writer.add_text(key, value, self.iteration)
            elif self.dtype[key] == 'embedding':
                self.writer.add_embedding(key, value, self.iteration)
            elif self.dtype[key] == 'pr_curve':
                self.writer.pr_curve(key, value['labels'], value['predictions'], self.iteration)
            elif self.dtype[key] == 'mesh':
                self.writer.add_audio(key, value, self.iteration)
            elif self.dtype[key] == 'hparams':
                self.writer.add_hparams(key, value['hparam_dict'], value['metric_dict'], self.iteration)
            else:
                raise Exception('Data type not supported, please update the visualizer plugin and rerun !!')

        self.iteration = self.iteration + 1