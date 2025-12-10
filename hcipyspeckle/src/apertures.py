import numpy as np
from scipy.ndimage import affine_transform
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp2d
from scipy.fftpack import fft2, ifft2
from scipy.linalg import norm
import multiprocessing as mp
from tqdm import tqdm  # For progress bar
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor, as_completed



def pupil_support_size(D, pixsize):
    """
    Get the next power of 2 larger than 2 * D/pixsize.

    """
    return 2 ** int(np.ceil(np.log2(2 * D / pixsize)))


def generate_aperture(diameter1, diameter2, pupil_width_pixels, pupil_sampling, normalize=True):

    x = np.array([[j for j in range(int(-pupil_width_pixels / 2), int(pupil_width_pixels / 2))] for _ in
                  range(pupil_width_pixels)]) * pupil_sampling
    aperture = ((diameter2 / 2) ** 2 <= (x ** 2 + x.T ** 2)) & ((x ** 2 + x.T ** 2) <= (diameter1 / 2) ** 2)
    aperture = aperture.astype(np.float32)

    if normalize:
        aperture /= norm(aperture)

    return aperture


def generate_aperture_chromatic_steps(N, λmin, λmax, delta_slice=1):
    """
    Generates a telescope aperture for multi-wavelength use.

    """
    rad_max = N // 4
    rad_min = int(np.ceil(rad_max * (λmin / λmax)))
    nwavs = int(np.round((rad_max - rad_min) / delta_slice) + 1)
    rad_min = rad_max - nwavs * delta_slice + delta_slice
    rad = np.zeros(nwavs, dtype=np.float32)
    λ = np.zeros(nwavs, dtype=np.float32)
    aperture_mask = np.zeros((N, N, nwavs), dtype=np.float32)

    for k in range(nwavs):
        rad[k] = rad_min + k * delta_slice
        λ[k] = λmin * (float(N) / 4.0) / rad[k]
        aperture_mask[:, :, k] = generate_aperture(2 * rad[k], 0, N, 1.0)

    return aperture_mask, λ


def generate_isoplanatic_frozen_flow_phase_extractors(atmosphere, timestamps, N, source_height, pupil_sampling,
                                                      dtype=np.float32):
    """
    Generate interpolators for isoplanatic frozen-flow phase extraction.

    """

    # Number of layers and wind vectors
    nlayers = atmosphere.nlayers
    n_winds = len(atmosphere.winds)

    if n_winds < nlayers:
        nlayers = n_winds  # Ensure nlayers matches wind vector count
        print(f"Warning: Number of layers exceeds number of wind vectors. Setting nlayers to the wind vector count ({nlayers}).")

    # Number of wavelengths and minimum wavelength
    nλ = len(atmosphere.λ)
    λmin = min(atmosphere.λ)
    nframes = len(timestamps)

    # Initialize the array to store interpolators for each layer, wavelength, and frame
    I = np.empty((nlayers, nλ, nframes), dtype=object)

    # Get the size of the phase screen
    Φ_size = atmosphere.phase_screens.shape[:2]  # This may not match N x N

    for n in range(nframes):
        for l in range(nλ):
            for i in range(nlayers):
                # Calculate the displacement in x and y directions based on the wind velocity and direction
                dx = (timestamps[n] * atmosphere.winds[i, 0] * np.cos(np.radians(atmosphere.winds[i, 1]))) / pupil_sampling
                dy = (timestamps[n] * atmosphere.winds[i, 0] * np.sin(np.radians(atmosphere.winds[i, 1]))) / pupil_sampling

                # Scaling factor to account for layer height and wavelength
                scaling = (1.0 - atmosphere.heights[i] / source_height) * (atmosphere.λ[l] / λmin)

                # Define the affine transformation matrix (scaling + translation)
                transform_matrix = np.array([[1 / scaling, 0, dx],
                                             [0, 1 / scaling, dy]])

                # Create a meshgrid for the phase screen
                grid_x, grid_y = np.meshgrid(np.arange(Φ_size[1]), np.arange(Φ_size[0]))

                # Apply the affine transformation to the grid
                transformed_grid_x = transform_matrix[0, 0] * grid_x + transform_matrix[0, 2]
                transformed_grid_y = transform_matrix[1, 1] * grid_y + transform_matrix[1, 2]

                # Flatten the transformed grids
                transformed_points = np.vstack([transformed_grid_y.ravel(), transformed_grid_x.ravel()]).T

                # Create an interpolator for each frame, layer, and wavelength
                interpolator = RegularGridInterpolator(
                    (np.arange(Φ_size[0]), np.arange(Φ_size[1])),
                    atmosphere.phase_screens[:, :, i],
                    method='linear',
                    bounds_error=False,
                    fill_value=0
                )

                # Interpolate the phase screens at the transformed grid points
                interpolated_values = interpolator(transformed_points).reshape(Φ_size)

                # Resize to match the output size N x N if necessary
                print(f"interpolated_values shape before ;{interpolated_values.shape}")
                if interpolated_values.shape != (N, N):
                    interpolated_values = resize(interpolated_values, (N, N), mode='reflect', anti_aliasing=True)

                # Store the interpolated values in the array I
                print(f"interpolated_values shape after ;{interpolated_values.shape}")
                I[i, l, n] = interpolated_values

    return I


def extract_composite_phases(Φ_layers, I, FTYPE=np.float32, verbose=True):
    """
    Monochromatic, isoplanatic, no propagation.
    """
    # Print shapes of input arrays
    print(f"Φ_layers shape: {Φ_layers.shape}")
    print(f"I shape: {I.shape}")

    nlayers, nλ, nframes = I.shape[0], I.shape[1], I.shape[2]
    pupil_width_pixels = I[0, 0, 0].shape[0]
    phases = np.zeros((pupil_width_pixels, pupil_width_pixels, nλ, nframes), dtype=FTYPE)

    if verbose:
        print("Computing composite phases")
        progress_bar = tqdm(total=nframes)

    # Check the range of frames
    def compute_phases(n):
        for l in range(nλ):
            # Debug the content of I and Φ_layers for any discrepancies
            try:
                interpolator = I[0, l, n]  # Check interpolator value
                print(f"Φ_layers shape: {Φ_layers.shape}")
                print(f"I shape: {I.shape}")
                print(f"Interpolator at I[0, {l}, {n}]: {interpolator.shape}")
                phase_layer = Φ_layers[:, :, 0, l]  # Check phase layer value
                print(f"Φ_layers[:, :, 0, {l}] shape: {phase_layer.shape}")

                phases[:, :, l, n] = np.sum([I[i, l, n] * Φ_layers[:, :, i, l] for i in range(nlayers)], axis=0)
            except Exception as e:
                print(f"Error while computing phase for frame {n}, layer {l}: {e}")

        if verbose:
            progress_bar.update(1)

    # Debug the range of frames
    for n in range(nframes):
        compute_phases(n)

    if verbose:
        progress_bar.close()

    return phases


def pupil_to_psf(A_pupil, Φ_pupil, psf_size, delta):
    """
    Convert pupil function to point spread function (PSF).

    """
    Np = 2 * Φ_pupil.shape[0]  # Pads out the array to avoid aliasing
    npatches = Φ_pupil.shape[2]
    nframes = Φ_pupil.shape[3]
    psfs = np.zeros((psf_size, psf_size, npatches, nframes), dtype=np.float64)

    def compute_psf(iframe):
        for ipos in range(npatches):
            padded = np.pad(A_pupil * np.exp(1j * Φ_pupil[:, :, ipos, iframe]),
                            ((0, Np - A_pupil.shape[0]), (0, Np - A_pupil.shape[1])), 'constant')
            psf = np.abs(fft2(padded)) ** 2
            psfs[:, :, ipos, iframe] = psf[:psf_size, :psf_size]  # Adjust to match PSF size

    # Use multiprocessing to parallelize the computation
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(compute_psf, range(nframes))

    return psfs