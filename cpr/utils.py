import os
import datetime
from typing import Dict, Any, Optional, List

import cv2
import yaml
import numpy as np
from onnxruntime import InferenceSession


def parse_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from yaml file.

    :param config_path: path to yaml file.
    :return: loaded dict.
    """
    with open(config_path) as file:
        parameters = yaml.load(file, Loader=yaml.SafeLoader)
    return parameters


def get_date() -> str:
    """
    Create string with current date and time.

    :return: created string.
    """
    t = datetime.datetime.now().timetuple()
    return '___{:02d}.{:02d}.{:02d}_{:02d}_{:02d}_{:02d}'.format(*[t[x] for x in [2, 1, 0, 3, 4, 5]])


class OnnxModelLoader:
    def __init__(self, onnx_path: str):
        """
        Class for loading ONNX models to inference on CPU.

        :param onnx_path: path to ONNX model file (*.onnx file).
        """
        self.onnx_path = onnx_path
        self.sess = InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])

        # In current case model always has exactly one input and one output.
        self.input_name = [x.name for x in self.sess.get_inputs()][0]
        self.output_name = [x.name for x in self.sess.get_outputs()][0]

    def inference(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run inference.

        :param inputs: input numpy array.
        :return: output numpy array.
        """
        outputs = self.sess.run([self.output_name], input_feed={self.input_name: np.float32(inputs)})
        return outputs[0]


def prepare_dataset(path: str, config_path: str) -> None:
    """
    Convert images to dataset and save it to .npy file. See "images" and "datasets" directories.
    This function MUST be modified to support adding new datasets.

    :param path: path to folder with folders with images sets. Each folder with images set contains folders for each
        class, these folders contain images.
    :param config_path: path to configuration file.
    """
    config = parse_yaml_config(config_path)
    cur_dataset = os.path.basename(path)
    if cur_dataset == 'tanks':
        train_name = 'T72 (train)'
        test_name = 'T72 (true)'
    elif cur_dataset == 'faces':
        train_name = 'Train'
        test_name = None
    elif cur_dataset == 'coil_20':
        train_name = 'Train'
        test_name = 'Test'
    elif cur_dataset == 'cars':
        train_name = '1_green'
        test_name = None
    elif cur_dataset == 'mnist_012':
        train_name = 'train_0'
        test_name = 'test_0'
    else:
        msg = 'Dataset "{}" is not supported yet.'.format(cur_dataset)
        raise TypeError(msg)
    datasets = [os.path.join(path, x) for x in os.listdir(path)]
    for dataset in datasets:
        data = {}
        false_counter = 1
        folders = [os.path.join(dataset, x) for x in os.listdir(dataset)]
        for folder in folders:
            if os.path.basename(folder) == train_name:
                key = 'train'
            elif os.path.basename(folder) == test_name:
                key = 'test'
            else:
                key = 'false_{}'.format(false_counter)
                false_counter += 1
            data[key] = []
            images = [os.path.join(folder, x) for x in os.listdir(folder)]
            for img in images:
                data[key].append(cv2.imread(img, 0))
            data[key] = np.stack(data[key], axis=0)
        name = os.path.join(config['basic']['data_path'], '{}_{}.npy'.format(cur_dataset, os.path.basename(dataset)))
        np.save(name, data)


def update_config(
    basic_config: Dict[str, Any],
    default_config: Dict[str, Any],
    *,
    input_slm_type: Optional[str] = None,
    filter_slm_type: Optional[str] = None,
    filter_type: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    save_data: Optional[bool] = None,
    input_slm: Optional[Dict[str, Any]] = None,
    filter_slm: Optional[Dict[str, Any]] = None,
    use_same_images_preprocessing: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Function for updating configuration dict. Use it always for updating modelling parameters.

    Raises ValueError if any parameter is wrong.

    :param basic_config: dict with default modelling parameters.
    :param default_config: dict with default modulator and post processing parameters.
    :param input_slm_type: type of the SLM for images.
    :param filter_slm_type: type of the SLM for correlation filters.
    :param filter_type: type of the correlation filter.
    :param metrics: list of used metrics.
    :param save_data: save all modelling results.
    :param input_slm: dict with parameters for the SLM for images.
    :param filter_slm: dict with parameters for the SLM for correlation filters.
    :param use_same_images_preprocessing: use same images preprocessing for recognized images and training images.
    :return: updated configuration dict.
    """

    _config = default_config
    new_config = {main_key: {key: _config[main_key][key] for key in _config[main_key]} for main_key in _config}

    if input_slm_type is not None:
        if input_slm_type not in basic_config['input_slm_types']:
            raise ValueError('Wrong input_slm_type: "{}".'.format(input_slm_type))
        new_config['input']['input_slm_type'] = input_slm_type

    if filter_slm_type not in basic_config['filter_slm_types']:
        raise ValueError('Wrong filter_slm_type: "{}".'.format(filter_slm_type))
    new_config['input']['filter_slm_type'] = filter_slm_type
    new_config['filter']['filter_slm_type'] = filter_slm_type
    new_config['post_processing']['filter_slm_type'] = filter_slm_type

    if filter_type is not None:
        if filter_type not in basic_config['filter_types']:
            raise ValueError('Wrong filter_type: "{}".'.format(filter_type))
        new_config['filter']['filter_type'] = filter_type

    if metrics is not None:
        for item in metrics:
            if item not in basic_config['available_metrics']:
                raise ValueError('Wrong metric: "{}".'.format(item))
        new_config['post_processing']['metrics'] = metrics

    if save_data is not None:
        new_config['post_processing']['save_data'] = save_data

    if input_slm is not None:
        new_config['input']['phase_depth'] = input_slm.get('phase_depth', 2.0)
        noise_type = input_slm.get('noise_type')
        if noise_type is not None:
            if noise_type not in basic_config['noise_types']:
                raise ValueError('Wrong noise_type: "{}".'.format(noise_type))
            new_config['input']['noise_type'] = noise_type
        new_config['input']['noise_level'] = input_slm.get('noise_level', 0.0)
        phase_type = input_slm.get('phase_type')
        if phase_type is not None:
            if phase_type not in basic_config['phase_types']:
                raise ValueError('Wrong phase_type: "{}".'.format(phase_type))
            new_config['input']['phase_type'] = phase_type

    if filter_slm is not None:
        new_config['filter']['sample_level'] = filter_slm.get('sample_level', 8)
        new_config['filter']['phase_depth'] = filter_slm.get('phase_depth', 2.0)
        noise_type = filter_slm.get('noise_type')
        if noise_type is not None:
            if noise_type not in basic_config['noise_types']:
                raise ValueError('Wrong noise_type: "{}".'.format(noise_type))
            new_config['filter']['noise_type'] = noise_type
        new_config['filter']['noise_level'] = filter_slm.get('noise_level', 0.0)
        phase_type = filter_slm.get('phase_type')
        if phase_type is not None:
            if phase_type not in basic_config['phase_types']:
                raise ValueError('Wrong phase_type: "{}".'.format(phase_type))
            new_config['filter']['phase_type'] = phase_type

    if use_same_images_preprocessing is not None:
        new_config['filter']['use_same_images_preprocessing'] = use_same_images_preprocessing

    return new_config
