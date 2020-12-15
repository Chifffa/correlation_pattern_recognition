import os

import cv2
import numpy as np

from cpr import amplitude_holo, ifft2, random_phase_mask

if __name__ == '__main__':
    # Read original image.
    examples_path = os.path.join('data', 'examples_for_github')
    image = cv2.imread(os.path.join(examples_path, 'lenna.bmp'), 0)

    # 1. Create Fourier hologram.
    holo = np.uint8(amplitude_holo(image, sample_level=8, scale=2))
    cv2.imwrite(os.path.join(examples_path, 'lenna_holo.bmp'), holo)

    # 2. Restore it.
    holo_restored = np.abs(ifft2(holo))
    holo_restored[holo_restored.shape[0] // 2, holo_restored.shape[1] // 2] = 0
    holo_restored = holo_restored - np.min(holo_restored)
    holo_restored = np.uint8(holo_restored / np.max(holo_restored) * 255)
    cv2.imwrite(os.path.join(examples_path, 'lenna_holo_restored_simple.bmp'), holo_restored)

    # 3. Create and restore Fourier hologram with phase mask.
    holo = amplitude_holo(random_phase_mask(image), sample_level=8, scale=2)
    cv2.imwrite(os.path.join(examples_path, 'lenna_holo_with_phase_mask.bmp'), np.uint8(holo))

    holo_restored = np.abs(ifft2(holo))
    holo_restored[holo_restored.shape[0] // 2, holo_restored.shape[1] // 2] = 0
    holo_restored = holo_restored - np.min(holo_restored)
    holo_restored = np.uint8(holo_restored / np.max(holo_restored) * 255)
    cv2.imwrite(os.path.join(examples_path, 'lenna_holo_restored_phase_mask.bmp'), holo_restored)

    # 4. Create and restore Fourier hologram with phase mask with special shape.
    holo = amplitude_holo(random_phase_mask(image), sample_level=8, scale=2, experiment_shape=(512, 1024))
    holo_restored = np.abs(ifft2(holo))
    holo_restored[holo_restored.shape[0] // 2, holo_restored.shape[1] // 2] = 0
    holo_restored = holo_restored - np.min(holo_restored)
    holo_restored = np.uint8(holo_restored / np.max(holo_restored) * 255)
    cv2.imwrite(os.path.join(examples_path, 'lenna_holo_restored_experiment.bmp'), holo_restored)
