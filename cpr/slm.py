from abc import abstractmethod
from typing import Dict, Any, Union

import numpy as np

from .correlation_filters import ot_mach, minace, sdf
from .cpr_utils import put_images_on_scene, amplitude_holo


class Modulator:
    def __init__(self, basic_config: Dict[str, Any]):
        """
        Base class for all modulators.

        :param basic_config: dict with default modelling parameters.
        """
        self.basic_config = basic_config

        self.fsf = basic_config['field_size_factor']
        self.max_pixel = basic_config['max_pixel']
        self.measured_phase_path = basic_config['measured_phase_path']
        self.default_sample_level = basic_config['default_sample_level']

        self.slm_type = None
        self.phase_depth = None
        self.noise_type = None
        self.noise_level = None
        self.phase_type = None

    @abstractmethod
    def __call__(self, images: Union[Dict[str, np.ndarray], np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Preprocessing images or preparing correlation filter using modulator parameters.

        :param images: dict with data.
        :return: dict with preprocessed images or prepared correlation filter.
        """
        pass

    @abstractmethod
    def set_from_config(self, config: Dict[str, Any]) -> None:
        """
        Update modulator parameters.

        :param config: dict with modelling parameters.
        """
        pass

    def amplitude_slm(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Amplitude SLM. The object displayed on the modulator is a two-dimensional matrix, consisting of positive
        numbers in the range [0; max_pixel].
        Amplitude noise does not change this range. Phase noise makes matrices complex.

        :param data: dict with data.
        :return: dict with processed data.
        """
        if self.noise_type is None:
            return data
        elif self.noise_type == 'amplitude':
            for obj in data:
                noise_level = int(self.noise_level * self.max_pixel)
                noise = np.random.randint(-noise_level, noise_level + 1, data[obj].shape)
                data[obj] = data[obj] + noise
                data[obj] = np.where(data[obj] < 0, 0, data[obj])
                data[obj] = np.where(data[obj] > self.max_pixel, self.max_pixel, data[obj])
            return data
        elif self.noise_type == 'phase':
            for obj in data:
                noise = np.random.randn(*data[obj].shape)
                noise = noise / np.max(np.abs(noise)) * self.noise_level
                data[obj] = data[obj] * np.exp(noise * 1j * 2 * np.pi)
            return data

    def phase_slm(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Phase SLM. The object displayed on the modulator is a two-dimensional matrix, consisting of complex numbers.

        :param data: dict with data.
        :return: dict with processed data.
        """
        if self.noise_type is None:
            return data
        elif self.noise_type == 'amplitude':
            for obj in data:
                noise = abs(np.random.randn(*data[obj].shape))
                noise = 1 - noise / np.max(noise) * self.noise_level
                data[obj] = data[obj] * noise
            return data
        elif self.noise_type == 'phase':
            for obj in data:
                noise = np.random.randn(*data[obj].shape)
                noise = noise / np.max(np.abs(noise)) * self.noise_level
                data[obj] = data[obj] * np.exp(noise * 1j * 2 * np.pi)
            return data

    def amplitude_phase_slm(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Amplitude-phase SLM. The object displayed on the modulator is a two-dimensional matrix, consisting of
        complex numbers.

        :param data: dict with data.
        :return: dict with processed data.
        """
        if self.phase_type == 'random':
            for obj in data:
                phase = np.random.randn(*data[obj].shape)
                phase = phase / np.max(np.abs(phase))
                data[obj] = data[obj] * np.exp(phase * 1j * self.phase_depth * np.pi)
            return data
        elif self.phase_type == 'linear':
            for obj in data:
                phase = data[obj] / self.max_pixel
                data[obj] = data[obj] * np.exp(phase * 1j * self.phase_depth * np.pi)
            return data
        elif self.phase_type == 'quadratic':
            for obj in data:
                phase = data[obj] ** 2 / (self.max_pixel ** 2)
                phase = np.exp(phase * 1j * self.phase_depth*np.pi)
                data[obj] = data[obj] * np.exp(phase * 1j * self.phase_depth * np.pi)
            return data
        elif self.phase_type == 'measured':
            phase = np.load(self.measured_phase_path)
            for obj in data:
                data[obj] = data[obj] * (1 + 0j)
                for i in range(data[obj].shape[0]):
                    for mm in range(data[obj].shape[1]):
                        for nn in range(data[obj].shape[2]):
                            _ind = int(data[obj][i, mm, nn].real)
                            data[obj][i, mm, nn] = data[obj][i, mm, nn] * phase[_ind]
            return data

    def dmd_slm(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Micromirror SLM (DMD). Not implemented yet.

        :param data: dict with data.
        :return: dict with processed data.
        """
        raise NotImplementedError('DMD Modulator is not implemented yet.')


class InputSLM(Modulator):
    def __init__(self, basic_config: Dict[str, Any]):
        """
        Modulator used to output images to be recognized.

        :param basic_config: dict with default modelling parameters.
        """
        super().__init__(basic_config)

        self.filter_slm_type = None

    def set_from_config(self, config: Dict[str, Any]) -> None:
        self.slm_type = config['input_slm_type']
        self.phase_depth = config['phase_depth']
        self.noise_type = config['noise_type']
        self.noise_level = config['noise_level']
        self.phase_type = config['phase_type']
        self.filter_slm_type = config['filter_slm_type']

    def __call__(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.filter_slm_type is not None:
            for obj in images:
                images[obj] = put_images_on_scene(images[obj], self.fsf, in_center=True)
        if self.slm_type == 'amplitude':
            images = self.amplitude_slm(images)
        elif self.slm_type == 'phase':
            images = self.phase_slm(images)
        elif self.slm_type == 'amplitude_phase':
            images = self.amplitude_phase_slm(images)
        elif self.slm_type == 'dmd':
            images = self.dmd_slm(images)
        # Converting to complex.
        for obj in images:
            images[obj] = images[obj].astype(np.complex)
        return images

    def phase_slm(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for obj in data:
            data[obj] = np.exp(data[obj] / self.max_pixel * np.pi * self.phase_depth * 1j)
        return super().phase_slm(data)

    def dmd_slm(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError('DMD Modulator is not implemented yet.')


class FilterSLM(Modulator):
    def __init__(self, basic_config: Dict[str, Any]):
        """
        Modulator used to output correlation filters.

        :param basic_config: dict with default modelling parameters.
        """
        super().__init__(basic_config)

        self.filter_type = None
        self.sample_level = None
        self.ot_mach_alpha = None
        self.ot_mach_beta = None
        self.minace_noise_level = None
        self.minace_nu = None

    def set_from_config(self, config: Dict[str, Any]) -> None:
        self.slm_type = config['filter_slm_type']
        self.phase_depth = config['phase_depth']
        self.noise_type = config['noise_type']
        self.noise_level = config['noise_level']
        self.phase_type = config['phase_type']
        self.filter_type = config['filter_type']
        self.sample_level = config['sample_level']
        self.ot_mach_alpha = config['ot_mach_alpha']
        self.ot_mach_beta = config['ot_mach_beta']
        self.minace_noise_level = config['minace_noise_level']
        self.minace_nu = config['minace_nu']

    def __call__(self, train_images: np.ndarray) -> np.ndarray:
        """
        Подготовка корреляционного фильтра.

        :param train_images: dict с тренировочными изображениями.
        :return: обработанный синтезированный корреляционный фильтр.
        """
        if self.filter_type == 'ot_mach':
            flt = ot_mach(train_images, self.ot_mach_alpha, self.ot_mach_beta)
        elif self.filter_type == 'minace':
            flt = minace(train_images, self.minace_noise_level, self.minace_nu)
        elif self.filter_type == 'sdf':
            flt = sdf(train_images, make_fft=True)
        else:
            msg = 'Correlation filter "{}" is not implemented now.'.format(self.filter_type)
            raise NotImplementedError(msg)
        if self.slm_type is None:
            return flt
        if self.slm_type in ('amplitude', 'amplitude_phase'):
            sample_level = self.sample_level
        else:
            sample_level = self.default_sample_level
        flt = amplitude_holo(flt, sample_level, self.fsf, is_correlation_filter=True)
        flt = {'filter': flt}
        if self.slm_type == 'amplitude':
            flt = self.amplitude_slm(flt)
        elif self.slm_type == 'phase':
            flt = self.phase_slm(flt)
        elif self.slm_type == 'amplitude_phase':
            flt['filter'] = np.expand_dims(flt['filter'], axis=0)
            flt = self.amplitude_phase_slm(flt)
            flt['filter'] = flt['filter'][0, :, :]
        elif self.slm_type == 'dmd':
            flt = self.dmd_slm(flt)
        return flt['filter']

    def phase_slm(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for obj in data:
            data[obj] = np.exp(data[obj] / (2 ** self.default_sample_level - 1) * np.pi * self.phase_depth * 1j)
        return super().phase_slm(data)

    def dmd_slm(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError('DMD Modulator is not implemented yet.')
