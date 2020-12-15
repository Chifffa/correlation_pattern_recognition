from typing import Tuple, Optional

import numpy as np


def put_images_on_scene(images: np.ndarray, scale: int, in_center: bool = True) -> np.ndarray:
    """
    Locate images on empty field.

    :param images: images array.
    :param scale: image enlargement factor.
    :param in_center: if True, then image will be in the center of empty field. Otherwise in the upper left quarter.
    :return: processed images.
    """
    shape = images.shape
    padding = [(0, 0)]
    if len(shape) == 2:
        shape = (1,) + shape
        padding = []
    if in_center:
        padding.append((shape[1] * (scale - 1) // 2, shape[1] * (scale - 1) // 2))
        padding.append((shape[2] * (scale - 1) // 2, shape[2] * (scale - 1) // 2))
    else:
        padding.append((int(shape[1] * (scale / 4 - 0.5)), int(shape[1] * (scale / 4 + 1.5))))
        padding.append((int(shape[2] * (scale / 4 - 0.5)), int(shape[2] * (scale / 4 + 1.5))))
    images = np.pad(images, pad_width=padding, mode='constant', constant_values=(0, 0))
    return images


def amplitude_holo(image: np.ndarray, sample_level: int, scale: int = 2, is_correlation_filter: bool = False,
                   experiment_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Creation of an amplitude Fourier hologram of an image.

    :param image: image or correlation filter.
    :param sample_level: sampling level (i.e. 8 for 256 grayscale levels).
    :param scale: image enlargement factor.
    :param is_correlation_filter: pass True if processing correlation filter, obtained in frequency plane.
    :param experiment_shape: (height, width) - size of the hologram for output to the modulator. If None,
        then synthesize the square hologram.
    :return: synthesized hologram or holographic correlation filter.
    """
    if is_correlation_filter:
        image = ifft2(image)
    if experiment_shape is not None:
        if image.shape[0] >= experiment_shape[0] or image.shape[1] >= experiment_shape[1]:
            raise ValueError('Experiment image shape MUST be greater or equal then original one.')
        new_image = np.zeros(experiment_shape, dtype=np.complex)
        dh = int(experiment_shape[0] / 4 - image.shape[0] // 2)
        dw = int(experiment_shape[1] / 4 - image.shape[1] // 2)
        new_image[dh:dh + image.shape[0], dw:dw + image.shape[1]] = image
        image = new_image
    else:
        image = put_images_on_scene(image, scale, in_center=False)
    image = np.real(fft2(image))
    image -= np.min(image)
    if not sample_level:
        return image
    else:
        image = (image / np.max(image) * (2 ** sample_level - 1)).astype(np.int)
        return image.astype(np.float)


def random_phase_mask(image: np.ndarray) -> np.ndarray:
    """
    Multiply Fourier hologram by a random phase mask to improve restoring quality.
    Random phase is "0" or "pi".

    :param image: hologram array.
    :return: processed array.
    """
    mask = np.random.randint(0, 2, image.shape)
    mask = np.where(mask == 1, 1, -1)
    return image * mask


def fft2(image: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """
    Direct 2D Fourier transform for the image.

    :param image: image array.
    :param axes: axes over which to compute the FFT.
    :return: Fourier transform of the image.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes)


def ifft2(image: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """
    Inverse 2D Fourier Transform for Image.

    :param image: image array.
    :param axes: axes over which to compute the FFT.
    :return: Fourier transform of the image.
    """
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image, axes=axes), axes=axes), axes=axes)


def correlate_2d(image: np.ndarray, flt: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """
    Calculation of 2D cross-correlation between the image and the correlation filter (obtained in frequency plane).
    Arrays must have equal dimension for correct calculation.
    Example: if you have image array with shape (100, 256, 256) and flt error with shape (256, 256) use np.expanddims
    to create flt with shape (1, 256, 256).

    :param image: image array.
    :param flt: correlation filter.
    :param axes: axes over which to compute the FFT.
    :return: cross-correlation matrix.
    """
    if len(image.shape) != len(flt.shape):
        msg = 'Arrays must have equal dimension. Got len(image.shape) = {}, len(flt.shape) = {}.'.format(
            len(image.shape), len(flt.shape)
        )
        raise ValueError(msg)
    return ifft2(fft2(image, axes=axes) * np.conj(flt), axes=axes)
