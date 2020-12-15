import os
import json
from typing import Tuple, Dict, Any, List

import numpy as np
from tqdm import tqdm

from .utils import get_date
from .slm import InputSLM, FilterSLM
from .post_processing import PostProcessing


class CorrelationPatternRecognition:
    def __init__(self, basic_config: Dict[str, Any]):
        """
        Modelling of correlation pattern recognition.

        :param basic_config: dict with default modelling parameters.
        """
        self.basic_config = basic_config
        self.data_path = basic_config['data_path']

        self.input_slm = InputSLM(basic_config)
        self.filter_slm = FilterSLM(basic_config)
        self.post_processing = PostProcessing(basic_config)

    def run_dataset(self, dataset: str, save_path: str, input_config: Dict[str, Any], filter_config: Dict[str, Any],
                    post_processing_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Run single modelling iteration on single dataset.

        :param dataset: path to prepared dataset in .npy format.
        :param save_path: path for saving results.
        :param input_config: dict with parameters for the SLM for images.
        :param filter_config: dict with parameters for the SLM for correlation filters.
        :param post_processing_config: dict with parameters for post processing correlation signals.
        :return: dict with error value for each metric.
        """
        post_processing_config['dataset_name'] = os.path.basename(dataset).split('.npy')[0]
        post_processing_config['save_path'] = save_path

        # Update modulators and post processing parameters.
        self.input_slm.set_from_config(input_config)
        self.filter_slm.set_from_config(filter_config)
        self.post_processing.set_from_config(post_processing_config)

        # Load dataset and make modelling iteration.
        images, train_images = self.load_images(dataset)
        images = self.input_slm(images)
        if filter_config['use_same_images_preprocessing']:
            self.input_slm.filter_slm_type = None
            train_images = self.input_slm({'0': train_images})['0']
        cf = self.filter_slm(train_images)
        result = self.post_processing(images, cf)
        return result

    def work(self, dataset: str, config_list: List[Dict[str, Any]]) -> None:
        """
        Make modelling for all configuration dicts, presented in config_list.

        :param dataset: dataset name from available_datasets or path to prepared dataset in .npy format.
        :param config_list: list with different configuration dicts.
        """
        if dataset == 'all':
            dataset = [os.path.join(self.data_path, x) for x in os.listdir(self.data_path)]
        elif dataset in self.basic_config['available_datasets']:
            dataset = [os.path.join(self.data_path, x) for x in os.listdir(self.data_path) if x.startswith(dataset)]
        else:
            dataset = [dataset]

        for i, config in tqdm(enumerate(config_list), desc='Configs'):
            save_path = os.path.join(self.basic_config['default_save_path'], 'Config_{:03d}{}'.format(i, get_date()))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            result = {m: [] for m in config['post_processing']['metrics']}

            # So that all results can be obtained at once for all CFs.
            if config['filter']['filter_type'] == 'all':
                correlation_filters = [_filter for _filter in self.basic_config['filter_types'] if _filter != 'all']
            else:
                correlation_filters = [config['filter']['filter_type']]

            for cor_flt in tqdm(correlation_filters, desc='Correlation filters'):
                config['filter']['filter_type'] = cor_flt
                for data in tqdm(dataset, desc='Dataset'):
                    r = self.run_dataset(data, save_path, config['input'], config['filter'], config['post_processing'])
                    for key in r:
                        result[key].append(r[key])

            # Correct entry in the final config so that all CFs could be used.
            if len(correlation_filters) > 1:
                config['filter']['filter_type'] = 'all'

            with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
                json.dump(config, f, indent=4)

            with open(os.path.join(save_path, 'result.txt'), 'w') as f:
                f.write('Datasets number = {}.\n'.format(len(dataset) * len(correlation_filters)))
                for key in result:
                    f.write('Metric "{}":\n'.format(key))
                    num_small_err = 0
                    num_big_err = 0
                    other = []
                    mean_error = []
                    for j in range(len(result[key])):
                        mean_error.append(np.clip(result[key][j], 0, 100))
                        if result[key][j] < 0.001:
                            num_small_err += 1
                        elif result[key][j] > 25:
                            num_big_err += 1
                        else:
                            other.append(result[key][j])
                    f.write('Number of sets with an error < 0.001% = {}.\n'.format(num_small_err))
                    f.write('Number of sets with an error > 25% = {}.\n'.format(num_big_err))
                    f.write('Mean error of other sets = {}.\n'.format(np.round(np.mean(other), 3)))
                    f.write('Mean error over all sets = {}.\n\n'.format(np.round(np.mean(mean_error), 3)))

    @staticmethod
    def load_images(path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Loading dataset.

        :param path: path to prepared dataset in .npy format.
        :return dict with images array for each object and images array for correlation filter synthesis.
        """
        try:
            images: Dict[str, np.ndarray] = np.load(path, allow_pickle=True).item()
            train_images = images['train'].copy()
        except Exception as e:
            print('Failed to download dataset "{}".'.format(path))
            raise e
        return images, train_images
