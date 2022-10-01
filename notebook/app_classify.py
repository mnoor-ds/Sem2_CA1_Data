# a simple app to classify images
# the following snippet was taken from Dash documentation

import datetime

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Learn Arabic'

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


def parse_contents(contents, filename):
    return html.Div([
        html.H5(filename),

        # placeholder for inference from a model
        html.H5('Arabic label:'),



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
