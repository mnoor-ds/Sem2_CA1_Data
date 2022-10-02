# a simple app to classify images
# the following snippet was taken from Dash documentation

#import datetime
#import io
from io import BytesIO
from PIL import Image
import numpy as np
import base64

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

from tensorflow.keras.models import load_model
from tensorflow import device as tf_device

import pickle
from itertools import islice


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Learn Arabic'


with tf_device('/cpu:0'):
    # in case GPU is being used for training, just load the model
    # on CPUs. run as CUDA_VISIBLE_DEVICES="" python app_classify.py
    # https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will/42750563#42750563
    keras_model = load_model('final_model')
    print('final_model loaded')


with open('name_id_map.pickle', 'rb') as handle:
    name_id_map = pickle.load(handle)

app.layout = html.Div([
    html.H2('Start Learning Arabic Today!'),
    html.P('This app can take multiple images and provide the name of ' + 
                'the objects in Arabic.'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def nth_key(dct, n):

    """
    a wrapper function to get item in a dictionary
    corresponding to the n-th key. this can be helpful
    in converting numpy.argmax result from TensorFlow
    prediction output to the corresponding class name

    source: https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model

    """

    it = iter(dct)
    # Consume n elements.
    next(islice(it, n, n), None) 
    # Return the value at the current position.
    # This raises StopIteration if n is beyond the limits.
    # Use next(it, None) to suppress that exception.
    return next(it)


def parse_contents(contents, filename):
        
    # convert the base64 encoded string into bytes
    # https://community.plotly.com/t/problem-when-converting-uploaded-image-in-base64-to-string-or-pil-image/38688/4
    encoded_image = contents.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    img = Image.open(bytes_image).convert('RGB')
    
    # convert the read-in bytes into numpy array
    img = np.array(img)
    # add another dimension as the model expects an input
    # in the shape of (a, b, c, d)
    img_expanded = np.expand_dims(img, axis=0)

    # run prediction and get the corresponding class name
    predict = keras_model.predict(img_expanded)
    predict = predict.argmax(axis=-1)[0]
    predict_str = nth_key(name_id_map, predict)
    # string manipulation to remove the parent folder name
    predict_str = predict_str.split('../full_data_split/train/')[-1]

    return html.Div([
        html.H5(filename),

        # placeholder for inference from a model
        html.H5('Arabic label: ' + predict_str),
        
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        

        #html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all' })
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              #State('upload-image', 'last_modified')
              )
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [parse_contents(c, n) for c, n in zip(
            list_of_contents, list_of_names)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
