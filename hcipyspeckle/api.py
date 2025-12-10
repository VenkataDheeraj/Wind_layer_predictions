import html
import os
import sys
import numpy as np
import h5py
import base64, subprocess
from flask import Flask, jsonify, send_file
import main

app = Flask(__name__)

@app.route("/")
def api_print():
    api_string = html.escape ("h5 endpoint: /v2/h5/wind/<0.0,3.0;...>/heights/<2400,...>/")
    api_string += "<br>"
    api_string += html.escape ("Metadata endpoint: /v2/metadata/wind/<0.0,3.0;...>/heights/<2400,...>/")
    return api_string

# /v2/metadata/wind/0.0,3.0;10.0,80.0;12.0,25.0;23.0,67.0/heights/2400,12400,22400,324/
@app.route("/v2/metadata/wind/<string:wind>/heights/<string:heights>/")
def metadata(wind, heights):

    return generate_data (False, wind, heights)

# /v2/h5/wind/0.0,3.0;10.0,80.0;12.0,25.0;23.0,67.0/heights/2400,12400,22400,324/
@app.route("/v2/h5/wind/<string:wind>/heights/<string:heights>/")
def h5(wind, heights):
    
    return generate_data (True, wind, heights)


# Shared Code
def generate_data (send_h5, wind = None, heights = None):
    commands = []

    if not heights is None:
       commands.append("--heights=" + heights)
    if not wind is None:
       commands.append("--winds=" + wind)

    h5_data = main.main_code(commands, web_api = True)

    if h5_data is None:
        return jsonify({'status':500, 'web_api_version':2.0, 'error':'No data generated'})
    
    #Load data
    metadata = {}
    metadata['status'] = '200'
    metadata['web_api_version'] = '2.0'
    metadata['diameter'] = h5_data['diameter']
    metadata['version'] = h5_data['version']
    metadata['pupil_min_sampling'] = h5_data['pupil_min_sampling']
    metadata['number_of_frames'] = h5_data['number_of_frames']
    metadata['exposure_time'] = h5_data['exposure_time']
    metadata['angular_distance_zenith'] = h5_data['angular_distance_zenith']
    metadata['elevation'] = h5_data['elevation']
    metadata['number_of_layers'] = h5_data['number_of_layers']
    metadata['propagation_distance'] = h5_data['propagation_distance']
    metadata['wind_vector'] = h5_data['wind_vector']
    metadata['wind_vector_dims'] = h5_data['wind_vector_dims']
    metadata['wind_vector_labels'] = h5_data['wind_vector_labels']
    metadata['location'] = h5_data['location']
    metadata['month'] = h5_data['month']
    metadata['coefficients'] = h5_data['coefficients']
    metadata['resolution'] = h5_data['resolution']
    metadata['min_observing_wavelengths'] = h5_data['min_observing_wavelengths']
    metadata['max_observing_wavelengths'] = h5_data['max_observing_wavelengths']
    metadata['timestamps'] = h5_data['timestamps']
    metadata['magnitude'] = h5_data['magnitude']
    metadata['pupil_support_size'] = h5_data['pupil_support_size']

    if send_h5:
        metadata['star_images'] = h5_data['star_images']
        metadata['zernike_coefficients'] =  h5_data['zernike_coefficients'].tolist()

    return jsonify(metadata)

if __name__ == '__main__':
    app.run(debug = True)
