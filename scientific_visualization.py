import os
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

from cpr import minace, fft2, ifft2, correlate_2d


def prepare_to_show(image_to_show: np.ndarray) -> np.ndarray:
    """
    Preparing image array to show and save.

    :param image_to_show: image to be processed.
    :return: processed image.
    """
    _image = np.abs(image_to_show.copy())
    _image -= np.min(_image)
    _image = _image / np.max(_image) * 255
    return _image.astype(np.uint8)


def get_minace(images_path: str) -> np.ndarray:
    """
    Create MINACE filter.

    :param images_path: path to images.
    :return: created MINACE filter.
    """
    _img_paths = [os.path.join(images_path, p) for p in os.listdir(images_path)]
    _images = [cv2.imread(p, 0) for p in _img_paths]
    _images = np.stack(_images, axis=0)
    return minace(_images)


def correlate_2d_images(image_1: np.ndarray, image_2: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """
    Calculation of 2D cross-correlation between images.
    Arrays must have equal dimension for correct calculation.
    Example: if you have image array with shape (100, 256, 256) and flt error with shape (256, 256) use np.expanddims
    to create flt with shape (1, 256, 256).

    :param image_1: first image array.
    :param image_2: second image array.
    :param axes: axes over which to compute the FFT.
    :return: cross-correlation matrix.
    """
    if len(image_1.shape) != len(image_2.shape):
        msg = 'Arrays must have equal dimension. Got len(image.shape) = {}, len(flt.shape) = {}.'.format(
            len(image_1.shape), len(image_2.shape)
        )
        raise ValueError(msg)
    return ifft2(fft2(image_1, axes=axes) * np.conj(fft2(image_2, axes=axes)), axes=axes)


def plot_3d(corr_matrix: np.ndarray, name: str) -> None:
    """
    Creates 3D plot of correlation signal.

    :param corr_matrix: correlation signal matrix.
    :param name: name of plot to be saved as png image.
    """
    x, y = np.mgrid[0:corr_matrix.shape[0], 0:corr_matrix.shape[1]]
    fig = plt.figure()
    ax = Axes3D(fig, rect=(0, 0, 1, 1), elev=30, azim=-60)
    surf = ax.plot_surface(
        x, y, corr_matrix, cmap=cm.get_cmap('copper'), linewidth=1, antialiased=False, rstride=1, cstride=1
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.set_size_inches(8, 5)
    fig.savefig(name + '_plot.png', dpi=300, bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    TRAIN_IMAGE_PATH = 'images/tanks/256_black/T72 (train)/T72_50_1_0225.jpg'
    TEST_TRUE_IMAGE_PATH = 'images/tanks/256_black/T72 (true)/T72_50_2_0222.jpg'
    TEST_FALSE_IMAGE_PATH = 'images/tanks/256_black/Abrams/abr_50_1_0225.jpg'
    MINACE_TRAIN_IMAGES_PATH = 'images/tanks/256_black/T72 (train)'

    # 0. Read images and create MINACE filter.
    train_image = cv2.imread(TRAIN_IMAGE_PATH, 0)
    test_true_image = cv2.imread(TEST_TRUE_IMAGE_PATH, 0)
    test_false_image = cv2.imread(TEST_FALSE_IMAGE_PATH, 0)
    minace_filter = get_minace(MINACE_TRAIN_IMAGES_PATH)
    minace_filter_image = prepare_to_show(minace_filter)

    # 1. Auto-correlation
    auto_corr_matrix = correlate_2d_images(train_image, train_image)
    auto_corr_matrix = prepare_to_show(auto_corr_matrix)
    plot_3d(auto_corr_matrix, '01_auto_corr')

    # 2. Cross-correlation (train and true).
    cross_corr_train_test_matrix = correlate_2d_images(train_image, test_true_image)
    cross_corr_train_test_matrix = prepare_to_show(cross_corr_train_test_matrix)
    plot_3d(cross_corr_train_test_matrix, '02_cross_corr_train_test')

    # 3. Cross-correlation (true and false).
    cross_corr_true_false_matrix = correlate_2d_images(test_true_image, test_false_image)
    cross_corr_true_false_matrix = prepare_to_show(cross_corr_true_false_matrix)
    plot_3d(cross_corr_true_false_matrix, '03_ross_corr_true_false')

    # 4. Cross-correlation (MINACE and true).
    cross_corr_minace_true_matrix = correlate_2d(test_true_image, minace_filter)
    cross_corr_minace_true_matrix = prepare_to_show(cross_corr_minace_true_matrix)
    plot_3d(cross_corr_minace_true_matrix, '04_cross_corr_minace_true_test')

    # 5. Cross-correlation (MINACE and false).
    cross_corr_minace_false_matrix = correlate_2d(test_false_image, minace_filter)
    cross_corr_minace_false_matrix = prepare_to_show(cross_corr_minace_false_matrix)
    plot_3d(cross_corr_minace_false_matrix, '05_cross_corr_minace_false_test')
