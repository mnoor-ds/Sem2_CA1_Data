# a simple app to classify images
# the following snippet was taken from Dash documentation

# import libraries to help with image loading
# and data preparation before running a prediction
from io import BytesIO
from PIL import Image
import numpy as np
import base64

# import dash components
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

# import modules from tensorflow to load models and 
# to specify 'attachment' of model to CPU
from tensorflow.keras.models import load_model
from tensorflow import device as tf_device

# import pickle to load a pickle object
import pickle
# import islice to help with iterable objects
from itertools import islice

# import an external CSS file for layout formatting
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# instantiate a Dash app with a good title
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Learn Arabic'


with tf_device('/cpu:0'):
    # in case there is no only GPU and it is being used for training,
    # just load the model on CPUs.
    # run as CUDA_VISIBLE_DEVICES="" python app_classify.py
    # https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will/42750563#42750563
    keras_model = load_model('final_model')
    print('final_model loaded')

# open the pickle file that maps sparse tensor of categories (e.g. [0, 1, 0...])
# to the corresponding label (e.g. 'blue' or 'black')
with open('name_id_map.pickle', 'rb') as handle:
    name_id_map = pickle.load(handle)

# create an HTML layout for a webpage
app.layout = html.Div([

    # add some decorative text
    html.H2('Start Learning Arabic Today!'),
    html.P('This app can take multiple images and provide the name of ' +
           'the objects in Arabic.'),

    # add a dash component to enable file uploads
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
        # allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])


def nth_key(dct, n):
    """
    a wrapper function to get an item in a dictionary
    corresponding to the n-th key. this can be helpful
    in converting numpy.argmax result from TensorFlow
    prediction output to the corresponding class name.

    source: https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model

    """

    # create an iterator object
    it = iter(dct)
    # consume n elements - this is essentially a loop
    # that starts at the n-th position and ends there.
    next(islice(it, n, n), None)

    # return the value at the current position.
    # this raises StopIteration if n is beyond the limits.
    # use next(it, None) to suppress that exception.
    return next(it)


def parse_contents(contents, filename):

    # convert the base64 encoded string into bytes
    # https://community.plotly.com/t/problem-when-converting-uploaded-image-in-base64-to-string-or-pil-image/38688/4
    # split the base64 encoded which has a little header text at the front ('data:image/image/png;base64,iVBORw...')
    encoded_image = contents.split(",")[1]
    # decode the base64 string into bytes
    decoded_image = base64.b64decode(encoded_image)
    # instantiate an I/O object with the bytes from above
    bytes_image = BytesIO(decoded_image)
    # use PIL to open the image and convert to RGB
    img = Image.open(bytes_image).convert('RGB')

    # convert the opened image into a numpy array
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

    html_div = html.Div([
                        # add a H5 heading to display the filename
                        html.H5(filename),

                        # string concatenation with a HTML H5 heading
                        # to format the prediction output
                        html.H5('Arabic label: ' + predict_str),

                        # HTML images accept base64 encoded strings in the same format
                        # that is supplied by the upload component
                        html.Img(src=contents),
                        # add a horizontal line to separate images if multiple files
                        # were uploaded
                        html.Hr(),
                        ])

    return html_div


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              )
def update_output(list_of_contents, list_of_names):
    """
    This function will be called upon file upload.

    'upload-image' is the component-id for the Dash upload object.
    Essentially, this function returns the contents (base64 encoded)
    and the file names once a user has uploaded image files.
    """

    if list_of_contents is not None:
        children = [parse_contents(c, n) for c, n in zip(
            list_of_contents, list_of_names)]
        return children


if __name__ == '__main__':
    # use debug=True to have visibility on errors etc.
    # note that the app will be run on a Flask server and is not 
    # suitable for live/production use.
    app.run_server(debug=False)
