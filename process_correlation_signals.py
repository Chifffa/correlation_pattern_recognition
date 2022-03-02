import os
from typing import Tuple, Optional

import cv2
import numpy as np

from cpr.utils import OnnxModelLoader


def get_maximum_coordinates(true_img_path: str) -> Tuple[Tuple[int, int], int]:
    """
    Get coordinates of correlation signal maximum and side of a square to cut.

    :param true_img_path: path to true correlation signal image.
    :return: (y, x) - coordinates; side of a square.
    """
    true_img = cv2.imread(true_img_path, 0)
    max_coords = np.where(true_img == np.max(true_img))
    max_y = int(np.median(max_coords[0]))
    max_x = int(np.median(max_coords[1]))
    square_side = min(max_y, true_img.shape[0] - max_y, max_x, true_img.shape[1] - max_x)
    return (max_y, max_x), square_side


def process_correlation(model_object: OnnxModelLoader, img_path: str, center: Tuple[int, int], square_side: int,
                        is_true: bool, stride: Optional[int], resize: Optional[int], save_path: str,
                        ) -> None:
    """
    Correlation signals processing.

    :param model_object: OnnxModelLoader object with loaded model.
    :param img_path: path to correlation signal image.
    :param center: coordinates of signal maximum around which to cut the square matrix (y_max, x_max).
    :param square_side: the side value of the square matrix.
    :param is_true: True for true signal or False for false signal.
    :param stride: use every i-th pixel of matrix.
    :param resize: cut square matrix of (resize, resize) shape around signal maximum.
    :param save_path: path to save obtained and processed correlation signals.
    """
    image = cv2.imread(img_path, 0)
    # Cut the square matrix around maximum coordinates.
    image = image[center[0] - square_side:center[0] + square_side, center[1] - square_side:center[1] + square_side]

    # Cut smaller square matrix around maximum coordinates.
    if resize is not None:
        h_center = image.shape[0] // 2
        w_center = image.shape[1] // 2
        image = image[h_center - resize // 2:h_center + resize // 2, w_center - resize // 2:w_center + resize // 2]

    # Select every i-th pixel of image.
    if stride is not None:
        image = image[::stride, ::stride]

    # Check image to be 32x32 (cut around center).
    image = image[image.shape[0] // 2 - 16:image.shape[0] // 2 + 16, image.shape[1] // 2 - 16:image.shape[1] // 2 + 16]

    # Don't forget to normalize image to be in [0, 1] and inference via trained model.
    image_to_process = image / np.max(image)
    # Returns [true_class_probability, false_class_probability]; true_class_probability + false_class_probability = 1.
    res = model_object.inference(image_to_process[np.newaxis, :, :, np.newaxis])[0]

    # Print and save results.
    if resize is None:
        resize = 768
    if stride is None:
        stride = 1
    if is_true:
        result = res[0]
    else:
        result = res[1]
    print('\nImage: "{}". Size: ({}, {}). Stride: {}. Class: {}. Predicted probability: {:.04f}%.\n'.format(
        os.path.basename(img_path), resize, resize, stride, is_true, result * 100
    ))
    cv2.imwrite(
        os.path.join(save_path, 'name-{}_size-{}_stride-{}_class-{}_predict-{}.png'.format(
            os.path.basename(img_path), resize, stride, is_true, result
        )),
        image
    )


if __name__ == '__main__':
    false_paths = [
        '/home/dmgoncharov/Downloads/False_1.png',
        '/home/dmgoncharov/Downloads/False_2.png'
    ]
    true_paths = [
        '/home/dmgoncharov/Downloads/True.png'
    ]
    saving_folder_path = '/home/dmgoncharov/Downloads/processed'
    strides_and_resizes = [(24, None), (16, 512), (8, 256), (4, 128), (2, 64), (None, 32)]

    model = OnnxModelLoader(os.path.join('data', 'frozen_model.onnx'))

    maximum_coordinates, square_side_value = get_maximum_coordinates(true_paths[0])
    os.makedirs(saving_folder_path, exist_ok=True)

    for (s, r) in strides_and_resizes:
        for p in false_paths:
            process_correlation(model, p, maximum_coordinates, square_side_value, False, s, r, saving_folder_path)
        for p in true_paths:
            process_correlation(model, p, maximum_coordinates, square_side_value, True, s, r, saving_folder_path)
