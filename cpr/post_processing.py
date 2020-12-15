import os
from typing import Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcdefaults, rcParams
from tqdm import tqdm

from .utils import get_date
from .cpr_utils import correlate_2d, ifft2
from .metrics import Peak, PSR, PCE, CNN


class PostProcessing:
    def __init__(self, basic_config: Dict[str, Any]):
        """
        Post processing of CPR modelling results.

        :param basic_config: dict with default modelling parameters.
        """
        self.basic_config = basic_config

        self.fsf = basic_config['field_size_factor']
        self.max_pixel = basic_config['max_pixel']
        self.measured_phase_path = basic_config['measured_phase_path']
        self.default_sample_level = basic_config['default_sample_level']
        self.model_path = basic_config['model_path']
        self.__built = False

        self.filter_slm_type = None
        self.dataset_name = None
        self.metrics = None
        self.save_data = None
        self.save_path = None
        self.date = None

    def set_from_config(self, config: Dict[str, Any]) -> None:
        """
        Update post processing parameters.

        :param config: dict with post processing parameters.
        """
        self.filter_slm_type = config['filter_slm_type']
        self.dataset_name = config['dataset_name']
        self.save_path = config['save_path']
        if not self.__built:
            self.metrics = []
            if 'peak' in config['metrics']:
                self.metrics.append(Peak(self.fsf))
            if 'psr' in config['metrics']:
                self.metrics.append(PSR(self.fsf))
            if 'pce' in config['metrics']:
                self.metrics.append(PCE(self.fsf))
            if 'cnn' in config['metrics']:
                self.metrics.append(CNN(self.model_path, self.fsf))
            self.save_data = config['save_data']
            self.__built = True
        else:
            for metric in self.metrics:
                metric.clear()

    def __call__(self, images: Dict[str, np.ndarray], flt: np.ndarray) -> Dict[str, float]:
        """
        Calculate correlations between images and correlation filter.

        :param images: dict with images arrays.
        :param flt: correlation filter array.
        :return: dict with errors calculated using different metrics.
        """
        self.date = get_date()
        for obj in tqdm(images, desc='Objects', leave=False):
            corr = np.zeros(images[obj].shape, dtype=complex)
            for i in tqdm(range(images[obj].shape[0]), desc='Images', leave=False):
                corr[i, :, :] = correlate_2d(images[obj][i, :, :], flt)
            if self.save_data:
                name = '{}_corr_{}_{}.corr'.format(self.dataset_name, obj, self.date)
                np.save(os.path.join(self.save_path, name), corr)
            for metric in self.metrics:
                metric.update(corr, obj, self.filter_slm_type)
        if self.save_data:
            self.save_all_data(flt)
        errors = {}
        for metric in self.metrics:
            error, threshold = metric.get()
            errors[metric.name] = error
            self.build_plot(metric.name, metric.values, threshold)
        return errors

    def save_all_data(self, flt: np.ndarray) -> None:
        """
        Save all data.

        :param flt: correlation filter array.
        """
        name = '{}_flt_{}.npy'.format(self.dataset_name, self.date)
        np.save(os.path.join(self.save_path, name), flt)
        for metric in self.metrics:
            np.save(os.path.join(self.save_path, name.replace('flt', metric.name)), metric.values)
        name = name.replace('npy', 'png')
        img = np.abs(flt) / np.max(np.abs(flt)) * self.max_pixel
        cv2.imwrite(os.path.join(self.save_path, name), img)
        img = np.abs(ifft2(flt))
        img = img / img.max() * self.max_pixel
        cv2.imwrite(os.path.join(self.save_path, name.replace('flt', 'flt_image')), img)

    def build_plot(self, metric_name: str, values: Dict[str, np.ndarray], threshold: float) -> None:
        """
        Build plots (discrimination characteristics).

        :param metric_name: current used metric name.
        :param values: dict with metric values per each object.
        :param threshold: recognition threshold value.
        """
        if metric_name == 'peak':
            metric_name = 'Correlation peak height'
        else:
            metric_name = metric_name.upper()
        rcdefaults()
        rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
        colors = ['black', 'navy', 'firebrick', 'seagreen', 'darkorange', 'yellow', 'cyan', 'blueviolet']
        ls = [(0, ()), (0, (3, 0.5)), (0, (3, 0.8, 1, 0.8)), (0, (1, 0.5)),
              (0, (2, 0.4, 1, 0.4, 1, 0.4)), (0, ()), (0, ()), (0, ())]
        legend = ['train']
        if 'test' in values:
            legend.append('test')
        for obj in values:
            if obj not in legend:
                legend.append(obj)
        max_len = max([len(values[obj]) for obj in values])
        fig = plt.figure(figsize=(18, 10), dpi=200)
        # Object plots.
        for i, obj in enumerate(legend):
            plt.plot(np.arange(1, len(values[obj]) + 1), values[obj], linestyle=ls[i], linewidth=5, color=colors[i])
        # Threshold line.
        plt.plot(np.arange(1, max_len + 1), np.zeros(max_len) + threshold,
                 color='thistle', linestyle=(0, (4, 1.2)), linewidth=5)
        plt.grid(color='lightgray', linestyle='--', linewidth=1)
        for i in range(len(legend)):
            legend[i] = legend[i].capitalize().replace('_', ' ')
        plt.legend(legend, bbox_to_anchor=(1.03, 0.62), loc=2, borderaxespad=0.1)
        plt.xlim(0, max_len)
        plt.ylim(0, 1.05)
        plt.xlabel('Image index')
        plt.ylabel(metric_name)
        name = '{}_{}_{}.png'.format(self.dataset_name, metric_name, self.date)
        fig.savefig(os.path.join(self.save_path, name),  bbox_inches='tight')
        plt.close('all')
