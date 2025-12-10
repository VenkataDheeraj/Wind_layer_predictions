
'''------------------------------------Dummy File----------------------------------'''
import h5py

# Path to your .h5 file
file_path = 'dataset/speckle_sat_n256_nframes1_nλ1_mag4_great_id_2024-09-11_20-22-03-1.h5'


# Open the file in read mode
with h5py.File(file_path, 'r') as h5_file:
    # List all datasets and groups within the file
    print("Keys in the file: %s" % h5_file.keys())

    dataset = h5_file['dataset']


    # Convert the dataset to a numpy array and print its contents
    data = dataset[:]

    print("Data in the dataset: ", data[0])

import matplotlib.pyplot as plt
for i in range(1):
    plt.imshow(data[i], cmap='gray')
    plt.show()


'''--------------------------------- Using hcipy --------------------------------------'''
'''--------------------------------- Using hcipy --------------------------------------'''
'''--------------------------------- Using hcipy --------------------------------------'''
'''--------------------------------- Using hcipy --------------------------------------'''
'''--------------------------------- Using hcipy --------------------------------------'''
import os
import sys
import numpy as np
import h5py
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from math import pi
from src.apertures import pupil_support_size
from hcipy import AtmosphericLayer, make_pupil_grid, make_circular_aperture, Wavefront, FraunhoferPropagator,FresnelPropagator,make_focal_grid_from_pupil_grid
from hcipy.atmosphere import InfiniteAtmosphericLayer
from src.zernikes import zernike
from src.generate_data import Detector, add_poisson_noise , convert_to_adu
from src.atmosphere import CN2_huffnagel_valley_generalized
from hcipy import Field
from skimage.transform import resize
from hcipy import read_fits,FastFourierTransform
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from hcipy.fourier import fourier_transform
from skimage.restoration import richardson_lucy
from scipy.signal import convolve2d
import os
import glob
import imageio.v2 as imageio

from src.atmosphere import CN2_huffnagel_valley




# Initialize Parameters
D = 360.0e-2
pixscale_wanted = 4e-2
λmin = 550.0e-9
λmax = 550.0e-9
nframes = 200
exptime = 10e-3
mag1 = 4
z = 17 / 360.0 * 2 * pi
elevation = 2400
location = "N/A"
month = "N/A"
coefficients = "0"
nlayers = 4
Dz = 30e3
layer_heights = [0.0]
set_heights = 0
object_distance = 35786e3
num_zernikes = 10
true_image = "./data/sat_template1.fits"
wind_file = " "
repeats = 1
thread_id = 1
save_path = "dataset/"
winds = np.array([[0.0, 3.0], [10.0, 80.0], [12.0, 25.0], [23.0, 67.0]])
winds_dim = 2

wind_file_index = 0
wind_file_array = []


def process_wind_file(wind_file_array):
    winds_param_array = []
    num_lines_per_block = 4  # Adjust based on the file structure
    for p in range(0, len(wind_file_array), num_lines_per_block):
        # Process only the second line in each 4-line block (the wind data)
        wind_line = wind_file_array[p + 1].strip()

        try:
            temp = np.array([list(map(float, pair.split(','))) for pair in wind_line.split(';')])

            if not np.isnan(temp).any():
                winds_param_array.append(temp)
            else:
                print(f"Warning: NaN detected in wind data at index {p // num_lines_per_block + 1}, skipping.")
        except Exception as e:
            print(f"Error processing wind data at index {p // num_lines_per_block + 1}: {e}")
            continue

    return winds_param_array


def print_help():
    print("Options")
    print("--diameter=? <default: 360.0e-2>")
    print("--frame-no=? <default: 2000>")
    print("--image=? <default: './data/sat_template1.fits'>")
    print("--pupil=? <default: 4e-2>")
    print("--zenith=? <default: 17/360.0*2*pi>")
    print("--layers=? <default: 4>")
    print("--heights=? <default: [2400,12400,22400,32400]")
    print("--winds=? <default: \"0.0,3.0;10.0,80.0;12.0,25.0;23.0,67.0\"")
    print("--exposure=? <default: 10e-3>")
    print("--magnitude=? <default: 4>")
    print("--elevation=? <default: 2400>")
    print("--location=? <default: N/A")
    print("--month=? <default: N/A")
    print("--coefficients=? <default: 0")
    print("--prop-distance=? <default: 30e3>")
    print("--object-distance=? <default: 35786e3>")
    print("--zernike=? <default: 10>")
    print("--export-dir=? <default: dataset/>")
    print("--wind-file=? ")
    print("--repeat=? <default: 1>")
    print("--thread-id=? <default: 1>")


def parse_args():
    global D, nframes, true_image, exptime, pixscale_wanted, z, mag1, nlayers, layer_heights, set_heights, winds, elevation
    global location, month, coefficients, Dz, object_distance, num_zernikes, save_path, wind_file, wind_file_index
    global wind_file_array, repeats, thread_id

    args = sys.argv[1:]
    for arg in args:
        if arg.lower() == "--help":
            print_help()
            sys.exit()

        key, value = arg.split("=")
        if key.lower() == "--diameter":
            D = float(value)
        elif key.lower() == "--frame-no":
            nframes = int(value)
        elif key.lower() == "--image":
            true_image = value
        elif key.lower() == "--exposure":
            exptime = float(value)
        elif key.lower() == "--pupil":
            pixscale_wanted = float(value)
        elif key.lower() == "--zenith":
            z = float(value)
        elif key.lower() == "--magnitude":
            mag1 = float(value)
        elif key.lower() == "--layers":
            nlayers = int(value)
        elif key.lower() == "--heights":
            layer_heights = list(map(float, value.strip("[]").split(",")))
            set_heights = 1
        elif key.lower() == "--winds":
            winds = np.array([[float(x) for x in item.split(",")] for item in value.split(";")])
            nlayers = winds.shape[0]
        elif key.lower() == "--elevation":
            elevation = float(value)
        elif key.lower() == "--location":
            location = value
        elif key.lower() == "--month":
            month = value
        elif key.lower() == "--coefficients":
            coefficients = value
        elif key.lower() == "--prop-distance":
            Dz = float(value)
        elif key.lower() == "--object-distance":
            object_distance = float(value)
        elif key.lower() == "--zernike":
            num_zernikes = int(value)
        elif key.lower() == "--export-dir":
            save_path = value
        elif key.lower() == "--wind-file":
            wind_file = value
            wind_file_index = 1
            with open(wind_file, 'r') as f:
                wind_file_array = f.readlines()
        elif key.lower() == "--repeat":
            repeats = int(value)
        elif key.lower() == "--thread-id":
            thread_id = int(value)



def smooth_aperture(N,aperture, pupil_grid, smoothing_factor=0.5):
    hann_window = np.hanning(N)[:, None] * np.hanning(N)[None, :]  # Hann window for smoothing
    hann_window = Field(hann_window.flatten(), pupil_grid)  # Create a Field compatible with the grid
    return aperture * hann_window

def deconvolve_image(observed_image, psf, num_iterations=30):
    return richardson_lucy(observed_image, psf, num_iter=num_iterations)

def generate_gif(X, T, process='forward', base_folder='gif_images'):

    for t in range(T):
        frame = X[t].reshape((256,256,1))
        imageio.imwrite(os.path.join(base_folder, f'frame_{t}.png'), ((X[t] * 255) // 1).astype(np.uint8))
    images = [0] * T

    for filename in glob.glob(f"gif_images/*.png"):
        idx = int(filename.split('/')[-1].split('_')[1].split('.')[0])
        if process == 'reverse':
            idx = T - idx - 1
        images[idx] = imageio.imread(filename)

    print(len(images))
    imageio.mimsave(f'{process}_process.gif', images)

    for image_file in glob.glob(f"gif_images/*.png"):
        os.remove(image_file)

def output_h5(run_l, layer_heights_l, winds_l, nlayers_l, location_l, month_l, coefficients_l, D, pixscale_wanted, λmin,
              λmax, nframes):
    # Setup grid and aperture
    N = pupil_support_size(D, pixscale_wanted)
    pupil_grid = make_pupil_grid(N, D)

    # Apply smooth aperture (ensure smooth edges)
    aperture = make_circular_aperture(D)
    aperture = aperture(pupil_grid)
    aperture = smooth_aperture(N, aperture, pupil_grid)  # Smooth the aperture for soft edges


    resolution = 0.25 * λmin * 1e6 / D / 2.0

    # Initialize timestamps
    timestamps = np.arange(nframes) * exptime
    detector = Detector(False, True, np.uint32, 1.0, 1.9, 2 ** 12 - 1, 0.0, exptime)

    # Set up object (generate image)
    object_image = read_fits(true_image)

    resize_factor = (pupil_grid.shape[0] / object_image.shape[0], pupil_grid.shape[1] / object_image.shape[1])
    object_image = zoom(object_image, resize_factor)
    #object_image = resize(object_image, (pupil_grid.shape[0], pupil_grid.shape[1]), anti_aliasing=True)

    cn2_profile = CN2_huffnagel_valley_generalized(layer_heights)
    L0 = 30
    # Generate atmosphere layers with proper turbulence and wind velocities
    layers = []
    for cn2_value, height,wind in zip(cn2_profile, layer_heights, winds):
        layer = InfiniteAtmosphericLayer(pupil_grid, cn2_value, L0=L0, velocity=wind)  # Example velocity
        layers.append(layer)


    distances = layer_heights_l
    fresnel_propagators = [FresnelPropagator(pupil_grid, d) for d in distances]

    # Simulate PSF for each timestamp
    fig, ax = plt.subplots(1, nframes, figsize=(15, 3))
    data = []
    deconvolved_images = []
    for i,t in enumerate(timestamps):
        # Create the wavefront for the aperture
        wavefront = Wavefront(aperture, λmin)
        object_wavefront = Wavefront(Field(object_image.flatten(), pupil_grid), λmin)
        #wavefront.electric_field *= object_wavefront.electric_field

        # Propagate through each atmospheric layer
        for layer, propagator in zip(layers, fresnel_propagators):
            layer.evolve_until(t)
            wavefront.electric_field *= np.exp(1j * layer.phase_for(wavefront.wavelength))
            propagated_wavefront = propagator.forward(wavefront)


        # Get the PSF after propagation
        #psf = wavefront.intensity.shaped
        psf = propagated_wavefront.intensity.shaped
        psf = psf / np.max(psf)  # Normalize PSF


        FTYPE = np.float32
        if detector.poisson:
            psf = detector.qe * add_poisson_noise(psf) + detector.sigma_ron * np.random.randn(N,N).astype(FTYPE)

    # Conversion to ADU if needed
        if detector.adu:
            psf = convert_to_adu(psf, detector)
        else:
            psf = np.maximum(psf, FTYPE(0))

        psf = np.array(psf)
        observed_image = convolve2d(object_image, psf, mode='same')


        # Save or visualize observed_image here
        data.append(psf)


        deconvolved_image = richardson_lucy(observed_image, psf, num_iter=30)
        deconvolved_images.append(deconvolved_image)
        print(f"image generated-{i}")

            # Visualize the result
        ax[i].imshow(data[i], cmap='gray')
        ax[i].imshow(deconvolved_images[i], cmap='gray')


    plt.suptitle(f' Image - Frame {t}')
    plt.show()

    generate_gif(deconvolved_images, nframes)




    plt.imshow(deconvolved_images[1]-deconvolved_images[0], cmap='gray')
    plt.title('Difference')
    plt.show()



    # Prepare Zernike coefficients for wavefront correction analysis
    z_polynomials = np.zeros((N, N, num_zernikes), dtype=np.float32)
    z_coefficients = np.zeros((num_zernikes, nframes), dtype=np.float32)

    for i in range(num_zernikes):
        z_polynomials[:, :, i] = zernike(i + 1, npix=N, diameter=N // 2, centered=True)

    electric_field_reshaped = wavefront.electric_field.reshape((N, N))

    for n in range(nframes):
        for i in range(num_zernikes):
            z_coefficients[i, n] = np.sum(z_polynomials[:, :, i] * np.angle(electric_field_reshaped)) / np.pi

    # Save data to HDF5 file
    savefile = f"speckle_sat_n{N}_nframes{nframes}_mag{mag1}_great_id_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{run_l}.h5"
    full_save_path = os.path.join(save_path, savefile)

    with h5py.File(full_save_path, 'w') as f:
        f.attrs['version'] = 1.3
        f.attrs['diameter'] = D
        f.attrs['pupil_min_sampling'] = pixscale_wanted
        f.attrs['number_of_frames'] = nframes
        f.attrs['exposure_time'] = exptime
        f.attrs['angular_distance_zenith'] = z
        f.attrs['elevation'] = elevation
        f.attrs['number_of_layers'] = nlayers_l
        f.attrs['propagation_distance'] = Dz
        f.attrs['wind_vector'] = ','.join(map(str, winds_l.flatten()))
        f.attrs['wind_vector_dims'] = winds_dim
        f.attrs['wind_vector_labels'] = "magnitude (m/s), angle (degrees)"
        f.attrs['location'] = location_l
        f.attrs['month'] = month_l
        f.attrs['coefficients'] = coefficients_l
        f.attrs['resolution'] = resolution
        f.attrs['min_observing_wavelengths'] = λmin
        f.attrs['max_observing_wavelengths'] = λmax
        f.attrs['timestamps'] = ','.join(map(str, timestamps))
        f.attrs['magnitude'] = mag1
        f.attrs['pupil_support_size'] = N

        # Save datasets
        f.create_dataset('psf_data', data=data, compression='gzip')
        f.create_dataset('zernike_coefficients', data=z_coefficients, compression='gzip')
        f.create_dataset('deconvolved_images', data=deconvolved_images, compression='gzip')

    print(f"File saved at {full_save_path}")


def process_profiles(x, wind_file_array, winds_param, winds_dim, D, pixscale_wanted, λmin, λmax, nframes, exptime,
                     true_image, mag1, z, object_distance, num_zernikes, save_path, elevation, Dz):
    print("process_profiles called")

    core_index = ((x - 1) * 4)  # Assuming each profile has 4 lines in the wind file
    layer_heights = np.array(list(map(float, wind_file_array[core_index].strip().split(','))))
    winds = winds_param[x - 1]  # Wind parameters for each layer
    nlayers = winds.shape[0]  # Number of atmospheric layers
    location = wind_file_array[core_index + 2].split(',')[0]  # Location (3rd line)
    month = wind_file_array[core_index + 2].split(',')[1]  # Month (3rd line)
    coefficients = wind_file_array[core_index + 3]  # Coefficients (4th line)

    # Run output_h5 with all layers combined in a single run
    output_h5(x, layer_heights, winds, nlayers, location, month, coefficients, D, pixscale_wanted, λmin, λmax, nframes)

    print("Finished saving output")


def main():
    parse_args()

    if set_heights == 0:
        global layer_heights
        layer_heights = elevation + np.arange(nlayers) * Dz / (nlayers - 1)

    # If we have a wind file, we process it and repeat the simulations
    if wind_file_index > 0:
        repeats = len(wind_file_array) // 4  # Assume 4 lines per profile in wind file
        print(f"Length of wind_file_array-{len(wind_file_array)}")

    if wind_file_index > 0:
        winds_param_array = process_wind_file(wind_file_array)
        print(f"Winds parameters: {winds_param_array}")

        # Use ProcessPoolExecutor to process multiple profiles concurrently
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(1, repeats + 1):  # Process each profile
                future = executor.submit(process_profiles, i, wind_file_array, winds_param_array, winds_dim, D,
                                         pixscale_wanted, λmin, λmax, nframes, exptime, true_image, mag1, z,
                                         object_distance, num_zernikes, save_path, elevation, Dz)
                futures.append(future)

            # Optionally, check results after all futures are done
            for future in futures:
                result = future.result()  # Wait for all tasks to complete
                print(f"Process completed with result: {result}")

    else:
        # If no wind file, just run a single profile
        output_h5(thread_id, layer_heights, winds, nlayers, location, month, coefficients, D, pixscale_wanted, λmin, λmax, nframes)


if __name__ == "__main__":
    main()