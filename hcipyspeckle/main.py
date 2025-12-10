from skimage.restoration import richardson_lucy

from src.apertures import pupil_support_size
import os
import sys
import numpy as np
import h5py
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from math import pi
from hcipy import  imshow_field,make_pupil_grid, make_circular_aperture, Wavefront, Cn_squared_from_fried_parameter, make_focal_grid,Field,read_fits,NoiselessDetector,Detector,make_elliptical_aperture,make_hexagonal_aperture,make_hexagonal_segmented_aperture,make_obstructed_circular_aperture,FraunhoferPropagator,make_fourier_transform
from hcipy.atmosphere import InfiniteAtmosphericLayer,MultiLayerAtmosphere,FresnelPropagator
from hcipy.mode_basis import noll_to_zernike, zernike, make_zernike_basis
from skimage.transform import resize
from hcipy.optics import NoisyDetector
from hcipy.field import make_focal_grid_from_pupil_grid
#from src.zernikes import zernike
import matplotlib.pyplot as plt
import glob
import imageio.v2 as imageio
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import time




# Initialize Parameters

D=1                                                     # Diameter of telescope aperture [m]#
pixscale_wanted = 4e-2                                  # [m/pix] pupil min sampling
λmin=550.0e-9                                           # min and max observing wavelengths [m]
λmax=550.0e-9
nframes = 2000                                          # Number of frames
exposure_time=0.005                                         # exposure time for each frame
mag1 = 4
z = 17/360.0*2*pi                                       # observation: angular distance from zenith [radians]
elevation = 2400                                        # observation: elevation
location = "N/A"                                        # location of wind profile
month = "N/A"                                           # month the wind profile was observed
coefficients="0"                                        # 16 values used as coefficients (comma columns, semi-color rows)
nlayers = 4                                             # number of atmospheric layers
Dz = 30e3                                               # propagation distance/elevation highest layer [m]
layer_heights = [0.0]                                   # Heights of each wind vector
set_heights = 0                                         # flag to determine if user set custom height
object_distance = 35786e3                               # [m] - closest distance to object
num_zernikes = 10                                       # Number of Zernike Co-Efficients
true_image = "./data/single_point.fits"
wind_file = " "                                         # TEMP: Load elevation and wind profile
repeats = 1                                             # How many times to repeat this process
thread_id = 1                                           # For multiple threads
save_path = "dataset/"                                  # Where files will be stored
focal_length = 10  # Focal length of telescope in meters
winds = np.array([[0.0, 3.0], [10.0, 80.0], [12.0, 25.0], [23.0, 67.0]]) ## (m/s, deg) 0deg = East, increases
winds_dim = 2
r0 = 0.17
L0 = 15.0
wind_file_index = 0
wind_file_array = []
fits_val = False
noise = False
max_workers = 8
# are we using the fits file or not


def process_wind_file(wind_file_array):
    winds_param_array = []
    num_lines_per_block = 4  # Adjust based on the file structure
    for p in range(0, len(wind_file_array), num_lines_per_block):
        # Process only the second line in each 4-line block (the wind data)
        wind_line = wind_file_array[p + 1]

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
    print("--max-workers=? <default: 64>")


def parse_args(custom_array = None):
    global D, nframes, true_image, exposure_time, pixscale_wanted, z, mag1, nlayers, layer_heights, set_heights, winds, elevation
    global location, month, coefficients, Dz, object_distance, num_zernikes, save_path, wind_file, wind_file_index
    global wind_file_array, repeats, thread_id

    args = custom_array
    if args is None:
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
            expo = float(value)
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
            print('wind_file:', wind_file)
            wind_file_index = 1
            with open(wind_file, 'r') as f:
                wind_file_array = f.readlines()
        elif key.lower() == "--repeat":
            repeats = int(value)
        elif key.lower() == "--thread-id":
            thread_id = int(value)
        elif key.lower() == "--max-workers":
            max_workers = int(value)


def crop_to_grid_size(image, grid_shape):
    """Crop the image to the specified grid shape centered at the middle."""
    y, x = image.shape
    new_y, new_x = grid_shape
    startx = x // 2 - (new_x // 2)
    starty = y // 2 - (new_y // 2)
    return image[starty:starty + new_y, startx:startx + new_x]


def generate_gif(X, T, process='forward', base_folder='gif_images'):
    X = np.array(X)
    min_val = X.min()
    max_val = X.max()
    X_norm = (X - min_val) / (max_val - min_val)
    for t in range(T):
        frame = X_norm[t].reshape((256,256, 1))
        cv2.imwrite(os.path.join(base_folder, f'frame_{t}.png'), ((frame * 255) // 1).astype(np.uint8))
    images = [0] * T

    for filename in glob.glob(f"gif_images/*.png"):
        idx = int(filename.split('/')[-1].split('_')[1].split('.')[0])
        if process == 'reverse':
            idx = T - idx - 1
        images[idx] = imageio.imread(filename)

    print(len(images))
    imageio.mimsave(f'gifs/{process}.gif', images)


    for image_file in glob.glob(f"gif_images/*.png"):
        os.remove(image_file)

def output_h5(run_l, layer_heights_l, winds_l, nlayers_l, location_l, month_l, coefficients_l, D, pixscale_wanted, λmin,
              λmax, nframes,fits=fits_val, noise =noise, from_api = False ):

    # dimensions of the frame generated
    N = pupil_support_size(D, pixscale_wanted)*4
    # print(N)

    # Makes a new Grid, meant for descretisation of a pupil-plane wavefront.
    # This grid is symmetric around the origin, and therefore has no point exactly on the origin for an even number of pixels.
    # make_pupil_grid(dims, diameter=1) https://docs.hcipy.org/0.3.1/api/hcipy.field.make_pupil_grid.html
    pupil_grid = make_pupil_grid(N, 2*D)
    # print(pupil_grid.shape)


    # Make a grid for a focal plane. https://docs.hcipy.org/dev/api/hcipy.field.make_focal_grid.html#make-focal-grid
    # make_focal_grid(q, num_airy, spatial_resolution=None, pupil_diameter=None, focal_length=None, f_number=None, reference_wavelength=None)
    focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, q=2, num_airy=N/2)
    # print('focal_grid',type(focal_grid))



    # Making aperture for different kinds of apertures
    # Example for circular aperture - circular_aperture(diameter, center=None)
    # https://docs.hcipy.org/dev/api/hcipy.aperture.circular_aperture.html#hcipy.aperture.circular_aperture
    aperture = make_circular_aperture(D)(pupil_grid)
    # Visualize the amplitude of the aperture
    # plt.figure(figsize=(6, 6))
    # imshow_field(aperture,  cmap='gray')
    # plt.colorbar()
    # plt.title("Aperture Field Amplitude")
    # plt.show()
    # print (type(aperture))


    # Step 1: Making a wavefront which represents the state of light to be propagated through the optical system
    # Example - Wavefront(electric_field, wavelength=1, input_stokes_vector=None)
    # https://docs.hcipy.org/dev/api/hcipy.optics.Wavefront.html#hcipy.optics.Wavefront
    wavefront = Wavefront(Field(np.ones(N*N), pupil_grid), λmin)

    # Combined amplitude and phase visualization
    # plt.figure(figsize=(12, 6))
    #
    # Amplitude subplot
    # plt.subplot(1, 2, 1)
    # imshow_field(amplitude, wavefront.grid,  cmap='gray')
    # plt.colorbar()
    # plt.title("Wavefront Amplitude")

    # Phase subplot
    # plt.subplot(1, 2, 2)
    # imshow_field(phase, wavefront.grid, cmap='gray')
    # plt.colorbar()
    # plt.title("Wavefront Phase")
    #
    # plt.show()

    # Cn_squared_from_fried_parameter https://docs.hcipy.org/dev/api/hcipy.atmosphere.Cn_squared_from_fried_parameter.html
    # Returns The integrated Cn^2 value for the atmosphere.
    cn2 = Cn_squared_from_fried_parameter(r0, λmin)

    # Resolution is used later for Zernike coefficients
    resolution = 0.25 * λmin * 1e6 / D / 2.0

    # Creating time steps based on number of frames and exposure time
    timestamps = np.arange(nframes) * exposure_time

    # Generate atmosphere layers
    layers = []
    for height, (speed, direction) in zip(layer_heights_l, winds_l):

        # Calculating Wind velocities by wind speed and direction
        vx = speed * np.cos(np.radians(direction))
        vy = speed * np.sin(np.radians(direction))

        #An atmospheric layer that can be infinitely extended in any direction. https://docs.hcipy.org/dev/api/hcipy.atmosphere.InfiniteAtmosphericLayer.html
        #InfiniteAtmosphericLayer(input_grid, Cn_squared, L0=inf, velocity=0, height=0, stencil_length=2, use_interpolation=True, seed=None)
        layer = InfiniteAtmosphericLayer(pupil_grid, cn2, L0, velocity=(vx, vy), height=height)
        layers.append(layer)

    # A multi-layer atmospheric model. https://docs.hcipy.org/dev/api/hcipy.atmosphere.MultiLayerAtmosphere.html
    # Step 2: MultiLayerAtmosphere(layers, scintillation=False, scintilation=True)
    atmosphere = MultiLayerAtmosphere(layers, scintillation=True)


    # Defining Detector
    # A detector class that has some basic noise properties.
    # https://docs.hcipy.org/0.5.1/api/hcipy.optics.NoisyDetector.html
    # NoisyDetector(detector_grid, dark_current_rate=0, read_noise=0, flat_field=0, include_photon_noise=True, subsampling=1)
    # changing these parameters gives weird outputs

    detect = NoisyDetector(
            detector_grid=pupil_grid,
            dark_current_rate = 0,
            flat_field =0,
            include_photon_noise=True,
            read_noise=0
        )






    # Reading the fits file by read_fits function
    # read_fits(filename, extension=0) it returns - ndarray
    object_image = read_fits(true_image)
    start_pos = int(N / 2)
    end_pos = int(3 * (2 * N) / 4)
    object_image = object_image[start_pos:end_pos, start_pos:end_pos]
    # print("object_image shape", object_image.shape)
    # object_image = object_image / np.max(object_image)

    # croping the image to desired dimensions
    # object_image = crop_to_grid_size(object_image, (N, N))

    # cv2.imwrite(os.path.join( f'object_image.png'), ((object_image * 255) // 1).astype(np.uint8))

    # Making a wavefront which represents the state of light to be propagated through the optical system
    # Example - Wavefront(electric_field, wavelength=1, input_stokes_vector=None)
    # https://docs.hcipy.org/dev/api/hcipy.optics.Wavefront.html#hcipy.optics.Wavefront
    # star_wavefront = Wavefront(Field(object_image.flatten(), pupil_grid), wavelength=λmin)

    # Create the Fourier transform operator
    # fourier_transform = make_fourier_transform(pupil_grid)

    data = []
    before_detector =[]
    wave_images = []
    star_images = []
    amp_images = []
    phase_images =[]
    for i, t in enumerate(timestamps):


        # Step 3:Evolve all atmospheric layers to a time t.
        # https://docs.hcipy.org/0.5.1/api/hcipy.atmosphere.MultiLayerAtmosphere.html#hcipy.atmosphere.MultiLayerAtmosphere.evolve_until
        atmosphere.evolve_until(t)

        # Step 4:Propagate the wavefront forward through the optical element.
        # https://docs.hcipy.org/0.5.1/api/hcipy.atmosphere.MultiLayerAtmosphere.html#hcipy.atmosphere.MultiLayerAtmosphere.evolve_until
        wavefront_after_atmosphere = atmosphere.forward(wavefront)



        # Step 5: Passing throught the aperture
        amplitude = (wavefront_after_atmosphere.amplitude * aperture).reshape(N, N)
        phase = (wavefront_after_atmosphere.phase * aperture).reshape(N, N)


        # This computes the intensity of the wavefront, it calculates the power or brightness distribution of the light in the focal plane. .shaped converts it into 2D
        # psf = wavefront_tmp.intensity.shaped
        # Step 6: Calculate the PSF intensity and normalize
        # psf_intensity = wavefront_tmp.intensity.shaped
        zero_arr = np.zeros((N, N), complex)
        zero_arr.real = amplitude * np.cos(phase)
        zero_arr.imag = amplitude * np.sin(phase)
        psf = np.abs(fftshift(fft2(zero_arr))) ** 2
        psf /= np.sum(psf)  # Normalize PSF to have a total sum of 1
        psf_fft = fft2(psf)


        phase_images.append(wavefront_after_atmosphere.phase)  # Storing Phase for visualization
        amp_images.append(wavefront_after_atmosphere.amplitude)
        data.append(np.sqrt(psf))


        # Step 7: Fourier Transform the object image
        image_fft = fft2(object_image)

        # Step 8: Convolve the object's Fourier spectrum with the PSF by multiplication
        convolved_fft = image_fft * psf_fft
        convolved_image = np.real(fftshift(ifft2((convolved_fft))))
        convolved_image /= convolved_image.max()

        # print(np.sum(convolved_image))

        before_detector.append(convolved_image)

        # Reading out the Wavefront using the detector
        # https://docs.hcipy.org/0.3.0/api/hcipy.optics.Detector.html#hcipy.optics.Detector.integrate
        # Integrate the detector with the convolved image
        # wavefront_convolved = Wavefront(Field(convolved_image.flatten(), focal_grid), wavelength=λmin)

        # detect.integrate(wavefront_convolved, exposure_time)
        # Read out the final image
        # detected_image = detect.read_out().shaped

        # Convert the intensity image to a Field for the detector
        intensity_field = Field(convolved_image.ravel(), pupil_grid)
        intensity_wavefront = Wavefront(intensity_field, λmin)
        intensity_wavefront.total_power = 10 ** 8  # Mag +8 StarLink Target in LEO

        detect.integrate(intensity_wavefront, exposure_time)
        # Read out the final image
        detected_image = detect.read_out().shaped
        # print(type(aperture))

        star_images.append(detected_image)

    # Save images and generate GIFs for each aperture type
    # generate_gif(data, nframes, process=f'PSF_image')
    # generate_gif(star_images, nframes, process=f'dtector_image')
    # generate_gif(phase_images, nframes, process=f'phase_image')
    # generate_gif(amp_images, nframes, process=f'amplitude_image')
    # generate_gif(before_detector, nframes, process=f'before_detector_image')


    # Prepare Zernike coefficients for wavefront correction analysis
    radial_z = []
    azimuthal_z = []
    z_polynomials = []
    
    for i in range (1, (num_zernikes + 1)):
        r, a = noll_to_zernike (i)
        radial_z.append (r)
        azimuthal_z.append (a)
    
    for i in range (num_zernikes):
        zern = zernike (radial_z[i], azimuthal_z[i], D, pupil_grid, True)
        z_polynomials.append (zern)
    
    #z_polynomials = np.zeros((N, N, num_zernikes), dtype=np.float32)
    z_coefficients = np.zeros((num_zernikes, nframes), dtype=np.float32)

    #for i in range(num_zernikes):
    #   z_polynomials[:, :, i] = zernike(i + 1, npix=N, diameter=N // 2, centered=True)

    #electric_field_reshaped = wavefront.electric_field.reshape((N, N))

    for n in range(nframes):
        for i in range(num_zernikes):
            z_coefficients[i, n] = np.dot (z_polynomials[i].reshape(N * N), phase_images[n].reshape(N * N)) / np.pi
            #z_coefficients[i, n] = np.sum(z_polynomials[:, :, i] * np.angle(electric_field_reshaped)) / np.pi

    # Save data to HDF5 file
    savefile = f"speckle_sat_n{N}_nframes{nframes}_mag{mag1}_great_id_{run_l}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"
    full_save_path = os.path.join(save_path, savefile)

    h5_data = {}

    if not from_api:
        with h5py.File(full_save_path, 'w') as f:
            f.attrs['version'] = 1.3
            f.attrs['diameter'] = D
            f.attrs['pupil_min_sampling'] = pixscale_wanted
            f.attrs['number_of_frames'] = nframes
            f.attrs['exposure_time'] = exposure_time
            f.attrs['angular_distance_zenith'] = z
            f.attrs['elevation'] = elevation
            f.attrs['number_of_layers'] = nlayers_l
            f.attrs['propagation_distance'] = Dz
            f.attrs['wind_vector'] = ','.join(map(str, winds_l.T.flatten()))
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
            f.create_dataset('star_images', data=star_images, compression='gzip')
            f.create_dataset('zernike_coefficients', data=z_coefficients, compression='gzip')
            # f.create_dataset('wave_images', data=wave_images, compression='gzip')
    else:
            h5_data['version'] = 1.3
            h5_data['diameter'] = D
            h5_data['pupil_min_sampling'] = pixscale_wanted
            h5_data['number_of_frames'] = nframes
            h5_data['exposure_time'] = exposure_time
            h5_data['angular_distance_zenith'] = z
            h5_data['elevation'] = elevation
            h5_data['number_of_layers'] = nlayers_l
            h5_data['propagation_distance'] = Dz
            h5_data['wind_vector'] = ','.join(map(str, winds_l.T.flatten()))
            h5_data['wind_vector_dims'] = winds_dim
            h5_data['wind_vector_labels'] = "magnitude (m/s), angle (degrees)"
            h5_data['location'] = location_l
            h5_data['month'] = month_l
            h5_data['coefficients'] = coefficients_l
            h5_data['resolution'] = resolution
            h5_data['min_observing_wavelengths'] = λmin
            h5_data['max_observing_wavelengths'] = λmax
            h5_data['timestamps'] = ','.join(map(str, timestamps))
            h5_data['magnitude'] = mag1
            h5_data['pupil_support_size'] = N

            for i in range (len (star_images)):
                star_images[i] = star_images[i].tolist()

            h5_data['star_images'] = star_images
            h5_data['zernike_coefficients'] = z_coefficients

    return h5_data, full_save_path


def process_profiles(x, wind_file_array, winds_param, winds_dim, D, pixscale_wanted, λmin, λmax, nframes, exposure_time,
                     true_image, mag1, z, object_distance, num_zernikes, save_path, elevation, Dz):
    # print("process_profiles called")

    core_index = ((x - 1) * 4)  # Assuming each profile has 4 lines in the wind file
    layer_heights = np.array(list(map(float, wind_file_array[core_index].strip().split(','))))
    winds = winds_param[x - 1]  # Wind parameters for each layer
    nlayers = winds.shape[0]  # Number of atmospheric layers
    location = wind_file_array[core_index + 2].split(',')[0]  # Location (3rd line)
    month = wind_file_array[core_index + 2].split(',')[1]  # Month (3rd line)
    coefficients = wind_file_array[core_index + 3]  # Coefficients (4th line)

    # Run output_h5 with all layers combined in a single run
    output_h5(x, layer_heights, winds, nlayers, location, month, coefficients, D, pixscale_wanted, λmin, λmax, nframes,fits=fits_val)
    
    print(f"Finished saving output for file id {x}")


def main():
    main_code (None)

# Can be called either from commandline or via API
def main_code(custom_array = None, web_api = False):
    parse_args(custom_array)

    if set_heights == 0:
        global layer_heights
        layer_heights = elevation + np.arange(nlayers) * Dz / (nlayers - 1)

    # If we have a wind file, we process it and repeat the simulations
    if wind_file_index > 0:
        repeats = len(wind_file_array) // 4  # Assume 4 lines per profile in wind file
        # print(f"Length of wind_file_array-{len(wind_file_array)}")

    if wind_file_index > 0:
        winds_param_array = process_wind_file(wind_file_array)
        # print(f"Winds parameters: {winds_param_array}")

        # Use ProcessPoolExecutor to process multiple profiles concurrently
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(1, repeats + 1):  # Process each profile
                future = executor.submit(process_profiles, i, wind_file_array, winds_param_array, winds_dim, D,
                                         pixscale_wanted, λmin, λmax, nframes, exposure_time, true_image, mag1, z,
                                         object_distance, num_zernikes, save_path, elevation, Dz)
                futures.append(future)

            # Optionally, check results after all futures are done
            for future in futures:
                result = future.result()  # Wait for all tasks to complete
                # print(f"Process completed with result: {result}")

    else:
        # If no wind file, just run a single profile
        data, output_file = output_h5(thread_id, layer_heights, winds, nlayers, location, month, coefficients, D, pixscale_wanted, λmin, λmax, nframes, fits = fits_val, from_api = web_api)
        if custom_array is None:
            print(output_file)
        else:
            return data

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()  # Run the main function
    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Time required to run the main function: {elapsed_time/60} minutes and {elapsed_time%60:.2f} seconds")
