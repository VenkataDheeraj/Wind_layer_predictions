import numpy as np
from math import factorial, sqrt, pi, exp, isclose
import warnings


# Stirling approximation for large numbers
def stirling(n):
    """
    Improved Stirling approximation for factorials.
    """
    if n < 20:
        return factorial(n)
    else:
        return sqrt(2 * pi * n) * (n / np.e) ** n * exp(1 / (12 * n) - 1 / (360 * n ** 3))


def noll_indexes(j):
    """
    Convert from 1-D to 2-D indexing for Zernikes or Hexikes.

    """
    if j < 1:
        raise ValueError("Zernike index j must be a positive integer")

    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n

    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def zernike_rad(n, m, rho):
    """
    Compute R[n, m], the Zernike radial polynomial.

    """
    m = abs(int(m))
    n = abs(int(n))
    output = np.zeros_like(rho)
    if (n - m) % 2 != 0:
        return output
    else:
        for k in range((n - m) // 2 + 1):
            output += (-1) ** k * stirling(n - k) / (
                    stirling(k) * stirling((n + m) // 2 - k) * stirling((n - m) // 2 - k)
            ) * rho ** (n - 2 * k)
        return output


def zernike(j, npix=128, diameter=128, cent_x=-1.0, cent_y=-1.0, outside=0.0, noll_normalize=True, centered=False):
    """
    Return the Zernike polynomial Z[j] for a given pupil.

    """
    if centered or cent_x == -1.0 or cent_y == -1.0:
        cent_x = npix // 2 + 1
        cent_y = npix // 2 + 1

    n, m = noll_indexes(j)
    x = (np.arange(1, npix + 1) - cent_x) / (diameter / 2.)
    y = (np.arange(1, npix + 1) - cent_y) / (diameter / 2.)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    aperture = np.ones_like(rho)
    aperture[rho > 1] = outside  # aperture mask
    norm_coeff = 1.0 / (diameter / 2)

    if noll_normalize:
        norm_coeff = sqrt(2 * (n + 1) / (1 + (m == 0))) / (diameter / 2)

    if m > 0:
        return norm_coeff * zernike_rad(n, m, rho) * np.cos(m * theta) * aperture
    elif m < 0:
        return norm_coeff * zernike_rad(n, m, rho) * np.sin(-m * theta) * aperture

    return norm_coeff * zernike_rad(n, 0, rho) * aperture


def circular_aperture(npix=128, diameter=128, cent_x=64.5, cent_y=64.5, outside=0.0, centered=False):
    """
    Returns a 2D aperture of the desired diameter pixels, centered on (cent_x, cent_y) on support npix X npix.

    """
    if centered:
        cent_x = npix // 2 + 1
        cent_y = npix // 2 + 1

    x = (np.arange(1, npix + 1) - cent_x) / (diameter / 2.)
    y = (np.arange(1, npix + 1) - cent_y) / (diameter / 2.)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    aperture = np.ones_like(rho)
    aperture[rho > 1] = outside  # aperture mask
    return aperture