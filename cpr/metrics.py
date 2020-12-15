from typing import Optional, Tuple

import numpy as np

from .utils import OnnxModelLoader


class Metric:
    def __init__(self, field_size_factor: int = 4):
        """
        Base class for all metrics.

        :param field_size_factor: image enlargement factor (if using Fourier holograms).
        """
        self.fsf = field_size_factor
        self.values = {}
        self.name = type(self).__name__.lower()

    def update(self, correlations: np.ndarray, obj: str, filter_slm_type: Optional[str]) -> np.ndarray:
        """
        Calculation of the metric using the matrix of correlation signals. When using holographic correlation filters,
        the region of interest is cut from this matrix.
        The method should be redefined and supplemented for all metrics.

        :param correlations: array with correlation signals matrices.
        :param obj: object name for which metric is being calculated.
        :param filter_slm_type: type of the modulator used to output correlation filters.
        """
        if filter_slm_type is None:
            corr = np.abs(correlations)
        else:
            height, width = correlations.shape[1:]
            h = int(height * (3 / 4 - 1 / self.fsf / 2))
            w = int(width * (3 / 4 - 1 / self.fsf / 2))
            corr = np.abs(correlations[:, h:h + height // self.fsf, w:w + width // self.fsf])
        return corr

    def get(self) -> Tuple[float, float]:
        """
        The method completes the metric calculation, returns error and recognize threshold value.

        :return: error and recognize threshold value.
        """
        max_value = max([max(self.values[obj]) for obj in self.values])
        for obj in self.values:
            self.values[obj] /= max_value
        if 'test' in self.values.keys():
            true = self.values['test']
        else:
            true = self.values['train']
        falses = {key: self.values[key] for key in self.values if key not in ['test', 'train']}
        false = falses[max(falses, key=lambda obj: np.mean(falses[obj]))]
        err, thresh = self.neumann_pearson(true, false)
        return err, thresh

    def clear(self) -> None:
        """
        Clearing all metrics results.
        """
        self.values = {}

    @staticmethod
    def neumann_pearson(true: np.ndarray, false: np.ndarray) -> Tuple[float, float]:
        """
        Neumann-Pearson criterion for finding recognition error in discriminatory characteristic.

        NOTE: old bad code, but works well.

        :param true: array with metric values for the true object.
        :param false: array with metric values for the false object.
        :return: error and threshold values, calculated by Neumann-Pearson criterion.
        """
        d = 0.0001
        x = np.arange(-1, 2, d)
        s = x.size

        def gauss(arg):
            mu = np.mean(arg)
            sigma = np.std(arg)
            if sigma < 1e-10:
                arg[0] = arg[0] - 0.0001
                mu = np.mean(arg)
                sigma = np.std(arg)
            g0 = 0
            g = np.zeros(s)
            for _i in range(s):
                g[_i] = np.exp(-0.5 * ((x[_i] - mu) / sigma) ** 2) / sigma / np.sqrt(2 * np.pi)
                g0 = g0 + g[_i] * d
            g = g / g0
            return g

        g_true = gauss(true)
        g_false = gauss(false)

        a = 10
        ind = g_false.argmax()
        rr = min(g_true.min(), g_false.min())
        m = g_true.argmax()
        for i in range(ind, int(2.0/d)):
            if abs(g_true[i]-g_false[i]) <= a:
                if abs(g_true[i]-g_false[i]) >= rr:
                    if i <= m:
                        a = abs(g_true[i]-g_false[i])
                        ind = i

        er1 = 0
        er2 = 0
        for i in range(0, ind):
            er1 = er1 + g_true[i+1]*d
        for i in range(ind, s-1):
            er2 = er2 + g_false[i+1]*d
        if er1 == 0:
            er1 = d*d
        if er2 == 0:
            er2 = d*d
        if a == 0:
            er1 = d*d
            er2 = d*d

        if x[ind] >= 0.95:
            return round(er1*100 + er2*100, 3), 0.95
        else:
            return round(er1*100 + er2*100, 3), x[ind]


class Peak(Metric):
    def __init__(self, field_size_factor: int = 4):
        """
        Metric "Normalized maximum height of the correlation signal".

        :param field_size_factor: image enlargement factor (if using Fourier holograms).
        """
        super().__init__(field_size_factor=field_size_factor)

    def update(self, correlations: np.ndarray, obj: str, filter_slm_type: Optional[str]) -> None:
        corr = super().update(correlations, obj, filter_slm_type)
        self.values[obj] = np.max(corr, axis=(1, 2))


class PSR(Metric):
    def __init__(self, field_size_factor: int = 4):
        """
        Metric "Peak to sidelobe ratio". Don't use it, because it's performance very bad. Maybe code is wrong.

        :param field_size_factor: image enlargement factor (if using Fourier holograms).
        """
        super().__init__(field_size_factor=field_size_factor)
        self.window_size = 2
        self.not_window_size = 0
        self.epsilon = 1e-6

    def update(self, correlations: np.ndarray, obj: str, filter_slm_type: Optional[str]) -> None:
        corr = super().update(correlations, obj, filter_slm_type)
        values = []
        for i in range(corr.shape[0]):
            ind_max = np.unravel_index(np.argmax(corr[i, :, :], axis=None), corr[i, :, :].shape)
            h_1 = int(np.clip(ind_max[0] - self.window_size, 0, corr.shape[1]))
            h_2 = int(np.clip(ind_max[0] + self.window_size, 0, corr.shape[1]))
            w_1 = int(np.clip(ind_max[1] - self.window_size, 0, corr.shape[2]))
            w_2 = int(np.clip(ind_max[1] + self.window_size, 0, corr.shape[2]))
            h_list = [h for h in range(h_1, h_2 + 1)
                      if not ind_max[0] - self.not_window_size <= h <= ind_max[0] + self.not_window_size]
            w_list = [w for w in range(w_1, w_2 + 1)
                      if not ind_max[1] - self.not_window_size <= w <= ind_max[1] + self.not_window_size]
            values.append((np.max(corr[i, :, :]) - np.mean(corr[i, h_list, w_list]))
                          / (self.epsilon + np.std(corr[i, h_list, w_list])))
        self.values[obj] = np.array(values)


class PCE(Metric):
    def __init__(self, field_size_factor: int = 4):
        """
        Metric "Peak to correlation energy ratio".

        :param field_size_factor: image enlargement factor (if using Fourier holograms).
        """
        super().__init__(field_size_factor=field_size_factor)

    def update(self, correlations: np.ndarray, obj: str, filter_slm_type: Optional[str]) -> None:
        corr = super().update(correlations, obj, filter_slm_type)
        self.values[obj] = (np.max(corr, axis=(1, 2)) ** 2) / np.sum(corr ** 2, axis=(1, 2))


class CNN(Metric):
    def __init__(self, model_path: str, field_size_factor: int = 4):
        """
        Metric "Convolutional neural network". It analyzes only the shape of signals and doesn't take into account
        signal's height.

        :param model_path: path to trained model, converted to onnx.
        :param field_size_factor: image enlargement factor (if using Fourier holograms).
        """
        super().__init__(field_size_factor=field_size_factor)
        self.model = OnnxModelLoader(model_path)

    def update(self, correlations: np.ndarray, obj: str, filter_slm_type: Optional[str]) -> None:
        corr = super().update(correlations, obj, filter_slm_type)
        corr = corr[:, corr.shape[1] // 2 - 16:corr.shape[1] // 2 + 16, corr.shape[2] // 2 - 16:corr.shape[2] // 2 + 16]
        results = []
        for i in range(corr.shape[0]):
            img = np.expand_dims(corr[i, :, :] / np.max(corr[i, :, :]), axis=-1)
            predicts = self.model.inference(np.expand_dims(img, axis=0))
            results.append(predicts[:, 0])
        self.values[obj] = np.concatenate(results, axis=0)

    def get(self) -> Tuple[float, float]:
        errors, length = 0, 0
        for key in self.values:
            if key in ('train', 'test'):
                errors += np.sum(np.where(self.values[key] <= 0.5, 1, 0))
            else:
                errors += np.sum(np.where(self.values[key] >= 0.5, 1, 0))
            length += len(self.values[key])
        return round(errors / length * 100, 3), 0.5
