import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
import os
import glob
import imageio.v2 as imageio
import cv2
import os
from hcipy import NoiselessDetector
from networkx.algorithms.approximation import diameter

from generate_data_main import layer_heights
import numpy as np

from hcipy import *

def generate_gif(X, T, process='forward', base_folder='gif_images'):
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)

    for t in range(T):
        frame = X[t].reshape((256, 256, 1))
        cv2.imwrite(os.path.join(base_folder, f'frame_{t}.png'), ((frame * 255) // 1).astype(np.uint8))
    images = [0] * T

    for filename in glob.glob(f"gif_images/*.png"):
        idx = int(filename.split('/')[-1].split('_')[1].split('.')[0])
        if process == 'reverse':
            idx = T - idx - 1
        images[idx] = imageio.imread(filename)

    imageio.mimsave(f'{process}_process.gif', images)

    for image_file in glob.glob(f"gif_images/*.png"):
        os.remove(image_file)





# Define parameters
wavelength = 500e-9  # 500 nm for visible light
diameter = 1.0  # Single aperture diameter in meters
focal_length = 10  # Focal length of telescope in meters

# Create a circular aperture for single aperture telescope
pupil_grid = make_pupil_grid(256, diameter)
aperture = circular_aperture(diameter)(pupil_grid)

# Set up atmospheric layers
layer1 = InfiniteAtmosphericLayer(pupil_grid, Cn_squared=1e-15, wind_speed=10, wind_direction=0)
layer2 = InfiniteAtmosphericLayer(pupil_grid, Cn_squared=5e-16, wind_speed=5, wind_direction=np.pi / 4)
atmosphere = MultiLayerAtmosphere([layer1, layer2])

# Define propagation system with atmosphere and telescope effects
prop = FraunhoferPropagator(pupil_grid, focal_length * wavelength / diameter)

# Create input field (e.g., from space) as a point source
input_field = Field(np.ones(pupil_grid.size), pupil_grid)

# Apply atmosphere effect by propagating through atmospheric layers
for layer in atmosphere.layers:
    input_field = layer(input_field, wavelength)

# Apply telescope aperture
input_field = aperture * input_field

# Propagate the field to the image plane to simulate final image
image_field = prop(input_field)




# Step 6: Visualize the results


# fig, ax = plt.subplots(1, nframes, figsize=(15, 3))
# for i in range(nframes):
#    ax[i].imshow(star_images[i], cmap='gray')
# plt.suptitle(f' Image - Frame {t}')
# plt.show()

# plt.imshow(star_images[0], cmap='gray')
# plt.title('Difference')
# plt.show()

# plt.imshow(abs(star_images[51] - star_images[0]), cmap='gray')
# plt.title('Difference')
# plt.show()
