import numpy as np

from .cpr_utils import fft2


def ot_mach(train_images: np.ndarray, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """
    OT MACH correlation filter.

    :param train_images: images array to synthesize filter.
    :param alpha: S matrix coefficient.
    :param beta: D matrix coefficient.
    :return: obtained OT MACH correlation filter.
    """
    img_number, height, width = train_images.shape

    x_matrix = fft2(train_images)
    x_matrix = np.reshape(x_matrix, (img_number, height * width))

    mean_vector = np.sum(x_matrix, axis=0) / img_number
    mean_vector = np.expand_dims(mean_vector, axis=0)

    s_matrix = np.sum((x_matrix - mean_vector) * np.conj(x_matrix - mean_vector), axis=0) / img_number

    d_matrix = np.sum(x_matrix * np.conj(x_matrix), axis=0) / img_number

    h_matrix = (alpha * s_matrix + beta * d_matrix) ** (-1)
    h_matrix *= mean_vector[0, :]
    h_matrix = np.reshape(h_matrix, (height, width))
    return h_matrix


def minace(train_images: np.ndarray, noise_level: int = 2, nu: float = 0.0) -> np.ndarray:
    """
    MINACE correlation filter.

    :param train_images: images array to synthesize filter.
    :param noise_level: noise level for noise matrix.
    :param nu: noise matrix coefficient.
    :return: obtained MINACE correlation filter.
    """
    img_number, height, width = train_images.shape

    x_matrix = fft2(train_images)
    x_matrix = np.reshape(x_matrix, (img_number, height * width))

    d_matrix = x_matrix * np.conj(x_matrix)

    noise_matrix = fft2(np.random.randint(1, noise_level, (height, width))) ** 2
    noise_matrix = np.reshape(noise_matrix, (1, height * width))

    t_matrix = np.max(np.concatenate([d_matrix, nu * noise_matrix], axis=0), axis=0)
    t_matrix = t_matrix ** (-1)

    # MINACE formula: H = T^(-1) * X * (X^(+)*T^(-1)*X)^(-1) * c.
    h_matrix = np.dot(x_matrix, np.conj(x_matrix.T) * np.expand_dims(t_matrix, axis=1))
    h_matrix = np.dot(np.linalg.inv(h_matrix), np.expand_dims(t_matrix, axis=0) * x_matrix)
    h_matrix = np.dot(np.ones(img_number), h_matrix)
    h_matrix = np.reshape(h_matrix, (height, width))

    return h_matrix


def sdf(train_images: np.ndarray, make_fft: bool = True) -> np.ndarray:
    """
    SDF correlation filter. Default synthesis is in object plane.

    :param train_images: images array to synthesize filter.
    :param make_fft: make Fourier transform for further filters compatibility.
    :return: obtained SDF correlation filter.
    """
    img_number, height, width = train_images.shape

    x_matrix = np.zeros((height * width, img_number))
    u = np.ones((img_number, 1))

    for i in range(img_number):
        x_matrix[:, i] = np.reshape(train_images[i, :, :], height * width)

    h = np.matmul(x_matrix, np.linalg.inv(np.matmul(x_matrix.transpose(), x_matrix)))
    h = np.matmul(h, u)
    h = np.reshape(h, (width, height))
    if make_fft:
        h = fft2(h)
    return h
