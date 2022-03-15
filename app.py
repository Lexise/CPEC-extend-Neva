import dash
import re
import time
import seaborn as sns
import base64
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import random
import io
from process_data import Process_data,get_color_label,get_colors
from multiprocessing import Process,Manager
from clingo_asp_compute import compute_extensions
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from urllib.parse import quote as urlquote
from file_manage import uploaded_files,get_current_processed_dir_semantic, save_file, clean_folder
from control import WELL_COLOR_new
from dash import dash_table
from ctypes import c_char_p
import plotly.graph_objects as go
from flask_caching import Cache
from flask import Flask, send_from_directory,request
from clustering_correlation import compute_serial_matrix,innovative_correlation_clustering,my_optimal_leaf_ordering,abs_optimal_leaf_ordering
import numpy as np
# from process_data import process_data,clean_folder,get_color_label, find_feature_group, process_data_two_sets,\
#     addional_process_individual, process_extension_individual,get_colors,find_semantic_files,get_catogery,initial_process_individual
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
import copy
from dash.dependencies import Input, Output, State, ClientsideFunction
from flask_caching import Cache
import pathlib
import os
APP_PATH = str(pathlib.Path(__file__).parent.resolve())   #include download
UPLOAD_DIRECTORY = APP_PATH+"/data/app_uploaded_files/"
PROCESSED_DIRECTORY=APP_PATH + "/data/processed/"
DEFAULT_DATA=APP_PATH + "/data/default_data/"
CACHE_DIRECTORY=APP_PATH+"/data/cache/"
FILE_LIST=""
EXTENSION_DIR=APP_PATH + "/data/extension_sets/"
ASP_DIR=APP_PATH+'/asp_encoding/'

ZIP_DIRECTORY=APP_PATH + "/data/processed_zip/"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    print("created")
if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)
    print("created")
if not os.path.exists(CACHE_DIRECTORY):
    os.makedirs(CACHE_DIRECTORY)
    print("created")
if not os.path.exists(ZIP_DIRECTORY):
    os.makedirs(ZIP_DIRECTORY)
    print("created")

#from subprocess import Popen, PIPE
# p = os.popen('ls -la')
# print(p.read())

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.scripts.config.serve_locally=True
app.css.config.serve_locally=True
server = app.server
cache_config = {
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR":CACHE_DIRECTORY,
}

# Empty cache directory before running the app
#if
clean_folder(CACHE_DIRECTORY)
# folder = os.path.join(APP_PATH, "data/cache/")
# for the_file in os.listdir(folder):
#     file_path = os.path.join(folder, the_file)
#     try:
#         if os.path.isfile(file_path):
#             os.unlink(file_path)
#     except Exception as e:
#         print(e)

@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    if os.path.exists(UPLOAD_DIRECTORY+path):
        return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)
    elif os.path.exists(PROCESSED_DIRECTORY+path):
        return send_from_directory(PROCESSED_DIRECTORY, path, as_attachment=True)
    elif os.path.exists(EXTENSION_DIR+path):
        return send_from_directory(EXTENSION_DIR, path, as_attachment=True)
    elif os.path.exists(ZIP_DIRECTORY + path):
        return send_from_directory(ZIP_DIRECTORY, path, as_attachment=True)
    else:
        return send_from_directory(DEFAULT_DATA, path, as_attachment=True)



@server.route("/hello")
def hello():
    """Serve a file from the upload directory."""
    return 'hello world'


# @server.route("/")
# def query_progress():
#     name = request.args.get('name')
#     print(name)
#     with open('/tmp/data.txt', 'r') as f:
#         data = f.read()
#         records = json.loads(data)
#         for record in records:
#             if record['name'] == name:
#                 return jsonify(record)
#         return jsonify({'error': 'data not found'})



app.config.suppress_callback_exceptions = True

mapbox_access_token = "pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w"
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=10, r=10, b=20, t=20),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    #title="",
    showlegend=True,
    titlefont= {"size": 32},
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=8,
    ),
    xaxis=dict(
            autorange=True,
            showgrid=False,
            ticks='',
            showticklabels=False
        ),
    yaxis=dict(
            autorange=True,
            showgrid=False,
            ticks='',
            showticklabels=False
        )
)


#dataset=pd.read_pickle('new_test.pkl')
# with open(UPLOAD_DIRECTORY+'long-island-railroad_20090825_0512.gml.20.apx', 'r') as file:
#     test = file.read()
# print(test)
# if  len(os.listdir(PROCESSED_DIRECTORY))==6:
#     loaded_processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")
# else:
#     loaded_processed_data=pd.read_pickle(DEFAULT_DATA + 'bar_data.pkl')
if  len(os.listdir(PROCESSED_DIRECTORY))==12:
    processed_semantics=get_current_processed_dir_semantic(PROCESSED_DIRECTORY)
elif len(os.listdir(PROCESSED_DIRECTORY))==6:
    processed_semantics=["combined"]
elif os.path.isfile(PROCESSED_DIRECTORY +"CombinedBar_data.pkl"):
    processed_semantics=["combined"]
else:
    processed_semantics=['stable','preferred']
dataset_all=pd.read_pickle(DEFAULT_DATA+'bar_data.pkl')
process_data = Process_data("start")
cache = Cache()
cache.init_app(app.server, config=cache_config)

TIMEOUT = 120
df = pd.DataFrame(
    {
        "First Name": ["Arthur", "Ford", "Zaphod", "Trillian"],
        "Last Name": ["Dent", "Prefect", "Beeblebrox", "Astra"],
    }
)




def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


def get_file_name(dir):
    files = uploaded_files(dir)
    if len(files) == 0:
        return [html.Li("")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]


argument_analysis=html.Div([

    dcc.Link(html.Button('back'), href='/'),
    html.Div(
        [
            html.Div(
                [
                    # html.H4(
                    #     "select range in histogram:",
                    #     className="control_label",
                    # ),
                    # dcc.RangeSlider(
                    #     id='my-range-slider',
                    #     min=0,
                    #     max=len(dataset_all),
                    #     step=1,
                    #     value=[5, int(0.5*len(dataset_all))]
                    # ),


                    html.P("Presented data:", style={"font-weight": "bold"},className="control_label"),
                    dcc.RadioItems(
                        id="data_present_selector",
                        options=[
                            {"label": "All ", "value": "all"},
                            {"label": "Interesting", "value": "interesting"},
                        ],
                        value="all",
                        labelStyle={"display": "inline-block"},
                        className="dcc_control",
                    ),

                    dcc.Checklist(
                        id="sort_selector",
                        options=[{"label": "descending order", "value": "decreased"}],
                        className="dcc_control",
                        value=["decreased"],
                    ),

                    html.Div([
                                    html.Span("Semantics:", style={"margin-top": "5%","font-weight": "bold"}),
                                    dcc.RadioItems(
                                        id="semantic-method-1-1",
                                        options=[
                                            {"label": x, "value": x} for x in processed_semantics

                                        ],
                                        labelStyle={"display": "inline-block"},
                                        value=processed_semantics[0],
                                    )],
                                    id="semantic-method-argument analysis",
                                    style={'marginLeft': '2%', 'width': '18%'},
                                ),

                    html.P("Cluster Algorithm:", style={"font-weight": "bold"}, className="control_label"),
                    dcc.RadioItems(
                        id="clustering-method",
                        options=[
                            {"label": "DBscan ", "value": "db"},
                            {"label": "Kmeans", "value": "km"},
                        ],
                        labelStyle={"display": "inline-block"},
                        value="db",
                        className="dcc_control",
                    ),


                    html.Div(
                        [html.H5(id="selected_cluster")],
                        id="selected argument",
                        className="dcc_control",
                        # className="mini_container",
                    ),
                    # html.Div(
                    #     id="card-1",
                    #     children=[
                    #
                    #         daq.LEDDisplay(
                    #             id="stable",
                    #             value="04",
                    #             color="#92e0d3",
                    #             backgroundColor="#FFFF",
                    #             size=50,
                    #         ),
                    #        daq.LEDDisplay(
                    #             id="prefer",
                    #             value="17",
                    #             color="#92e0d3",
                    #             backgroundColor="#FFFF",
                    #             size=50,
                    #         ),
                    #     ],
                    #     className="row container-display",
                    # ),



                    html.Div(
                        #[
                            # html.Div(
                            #     [html.H6(id="stable"), html.P("Stable")],
                            #     id="stable_block",
                            #     className="mini_container",
                            # ),
                            # html.Div(
                            #     [html.H6(id="prefer"), html.P("Preferred")],
                            #     id="prefer_block",
                            #     className="mini_container",
                            #
                            # ),
                            # html.Div(
                            #     [html.H6(id="stage"), html.P("Stage")],
                            #     id="stage_block",
                            #     className="mini_container",
                            # ),

                        #],
                        id="info-container",
                        className="row container-display"
                    )

                ],
                className="pretty_container four columns",
                id="cross-filter-options",
            ),
            html.Div([dcc.Graph(id="bar_chart"),
                      dcc.RangeSlider(
                          id='my-range-slider',
                          min=0,
                          max=len(dataset_all),
                          step=1,
                          value=[int(0.2 * len(dataset_all)), int(0.5 * len(dataset_all))]
                      ),
                      ],
                     className="pretty_container seven columns",
                     style={'width': '64%'}),

        ],

        className="row flex-display",
    ),
    html.Div([
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="basic-interactions"),
        )
    ],
    className="row",
    ),
    html.Div([
        html.Div([
        dcc.Graph(
            id='basic-interactions'),
        dcc.RadioItems(
                id="argument-dimensional-reduction",
                options=[
                    {"label": "Tsne ", "value": "tsne"},
                    {"label": "SVD", "value": "svd"},
                    {"label": "AutoEncode", "value": "auto"},
                ],
                value="tsne",
                labelStyle={"display": "inline-block"},
                className="dcc_control",
            ),
        ],
        className="pretty_container seven columns"
        ),

        html.Div(
            [dcc.Graph(id="pie_graph")],
            className="pretty_container five columns",
        ),



    ],
    className = "row flex-display",
    ),
    # html.Div(
    #     [
    #         html.Br(),

    #     className="row flex-display",
    # ),
],

)

correlation_page= html.Div([
                dcc.Link(html.Button('back'), href='/'),
                html.Div([

                dcc.Graph(
                    id="correlation_hm"
                   ),

#html.Button('Correlation Matrix',style={'marginLeft': '2%', 'width': '49%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}),
                html.Div([
                    html.Button('HRP',
                                style={'font-size': '14px','marginLeft': '2%', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                                id='btn-nclicks-1', n_clicks=0),
                    html.Button('Revised HRP',
                                style={'font-size': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                                id='btn-nclicks-2', n_clicks=0),
                    html.Button('OLO',
                                style={'font-size': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                                id='btn-nclicks-3', n_clicks=0),
                    html.Button('Revised OLO',
                                style={'font-size': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                                id='btn-nclicks-4', n_clicks=0),
                    # html.P("Presented semantic extension:", style={"font-weight": "bold"}, className="dcc_control"),
                    #
                    # dcc.RadioItems(
                    #     id="data_semantic_correlation",
                    #     loading_state={"is_loading":True},
                    #     options=[
                    #         {"label": "Preferred ", "value": "pr"},
                    #         {"label": "Stage", "value": "stg"},
                    #     ],
                    #     value="pr",
                    #     labelStyle={"display": "inline-block"},
                    #     className="dcc_control",
                    # ),
                    ],
                className="row flex-display"
                )

                ],
            className="pretty_container")
    ])

main_page =     html.Div([


        # empty Div to trigger javascript file for graph resizing
    html.Div(id="output-clientside"),
    html.Div([
        dcc.Dropdown(
            id='visual-dropdown',
            options=[
                {'label': 'new processing', 'value': 'upload-user'},
                {'label': 'two dimension', 'value': 'dropdown-2d'},
                {'label': 'three dimension', 'value': 'dropdown-3d'},

            ],
            value='upload-user',
            style={'font-size': '14px', 'width': '150px','marginLeft': '2%'}
        ),

        #html.Br(),
        dcc.Link(html.Button('Argument Analysis',
                             style={'marginLeft': '4%','marginRight': '8px','width': '300px', 'font-size': '14px', "color": "#FFFF", "backgroundColor": "#2F8FD2"}),
                 href='/page-argument'),

        # dcc.Link(html.Button('3D Analysis',style={'marginLeft': '2%','width': '32%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-3d'),
        dcc.Link(html.Button('Correlation Matrix',
                             style={'marginLeft': '4%', 'width': '300px', 'font-size': '14px', "color": "#FFFF",
                                    "backgroundColor": "#2F8FD2"}), href='/page-correlation'),
    #html.Hr(),
    ],
    className="row flex-display",
    ),




    html.Div([
                    html.Div([
                    dcc.Graph(
                        id="scatter_groups",style={'display':'none'})
                    ],
                    className="bare_container"),




                    html.Div([
                    dcc.Graph(id="3d_scatter_cluster",className="row flex-display",style={'display':'none'}),

                    dcc.Graph(id="3d_scatter_group", className="row flex-display",style={'display':'none'}),
                    ],

                    className="row flex-display",
                    style={'border-radius': '5px',
                          'margin': '12px',
                          'padding': '17px'}
                ),



        ], className="row flex-display",
           # style={'margin':dict(l=10, r=0, b=0, t=0)}
        # parent_style={'flex-direction': 'column',
        #                            '-webkit-flex-direction': 'column',
        #                            '-ms-flex-direction': 'column',
        #                           'display': 'flex'},
        #vertical = True

        ),



        html.Div([
            html.Div([
                    html.Span("Semantics:", style={"margin-top": "5%","font-weight": "bold"}),
                    dcc.RadioItems(
                        id="semantic-method-1",
                        options=[
                            {"label": x, "value": x} for x in processed_semantics
                        ],
                        labelStyle={"display": "inline-block"},
                        value=processed_semantics[0],
                    )],
                    id="semantic-method",
                    style={'marginLeft': '2%', 'width': '18%'},
                            ),
            html.Div(children=[
                html.Span("Dimensional Reduction:", style={"font-weight": "bold"}),
                dcc.RadioItems(
                    id="dimensional-reduction1",
                    options=[
                        {"label": "Tsne ", "value": "tsne"},
                        {"label": "SVD", "value": "svd"},
                        {"label": "AutoEncode", "value": "auto"},
                    ],
                    labelStyle={"display": "inline-block"},
                    value="tsne",
                ),
            ],
                style={'marginLeft': '2%', 'width': '18%'},

            ),

            html.Div(
                     [
                html.Span("Cluster Algorithm:", style={"font-weight": "bold"}),
                dcc.RadioItems(
                    id="clustering-method",
                    options=[
                        {"label": "DBscan ", "value": "db"},
                        {"label": "Kmeans", "value": "km"},
                    ],
                    labelStyle={"display": "inline-block"},
                    value="db",
                )],

                style={'marginLeft': '2%', 'width': '18%'},
            ),

            html.Button(id='feature_semantic', children='semantics identifier',className='middle-button',
                        style={'marginTop':'0.5%','font-size': '14px', "color": "#000000", "backgroundColor": "#f0f0f0"}),
            dbc.Tooltip(

                id='feature_semantic_table',
                target="feature_semantic",
                style ={'font-size': '11px',"color": "#000000", "backgroundColor": "#f0f0f0"}
            ),
            html.Button(id='feature_cluster', children='clusters identifier',className='middle-button',
                        style={'marginTop':'0.5%','font-size': '14px','marginLeft': '3%', "color": "#000000", "backgroundColor": "#f0f0f0"}),#'marginTop':'1%',

            dbc.Tooltip(
                id='feature_cluster_table',
                target="feature_cluster",
                style={'font-size': '11px', "color": "#000000", "backgroundColor": "#f0f0f0"}
            ),



        ],
            id='options-visualization',
            style={'display':'none'},
            className="row flex-display"
        ),

        html.Div(id='hover-data'),

        # html.Br(),
        # dcc.Link(html.Button('Argument Analysis',style={'width': '32%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-argument'),
        # #dcc.Link(html.Button('3D Analysis',style={'marginLeft': '2%','width': '32%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-3d'),
        # dcc.Link(html.Button('Correlation Matrix',style={'marginLeft': '2%', 'width': '32%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-correlation'),
        # html.Hr(),
        html.Div([
        dcc.Tabs([
#html.Br(),
            dcc.Tab(label='UPLOAD', children=[
                html.Div(
                    [
                    html.Div([
                        dcc.Upload([html.Button(children='UPLOAD APX', id="upload_button")], id="upload-data",
                                   style={'width': '13%', 'marginLeft': '16%', 'marginTop': '1%', 'font-size': '15px',
                                          'font-weight': 'bold', "color": "#FFFF",'marginRight': '10%'}, multiple=True),
                        dcc.Upload([html.Button(children='UPLOAD Extensions', id="upload_ex_button")], id="upload-ex-data",
                                   style={'width': '13%', 'marginLeft': '15%', 'marginTop': '1%', 'font-size': '15px',
                                          'font-weight': 'bold', "color": "#FFFF"}, multiple=True),
                    ],
                    className = "row flex-display"),
                    html.Div([


                        html.Div([html.P("Uploaded",className="dcc_control",style={"width":"80%",'textAlign': 'left'}),
                        html.Button(id='clear-upload', n_clicks=0, children='Clear',style={'font-size':'11px','textAlign': 'left','marginRight': '1%'}),
                        ],
                        className = "row flex-display"   ),

                        html.Ul(id="file-list", children=get_file_name(UPLOAD_DIRECTORY))
                        ],
                    className="pretty_container",),

                    #style={'border-radius': '15px'}),
                     ],
                className ='empty_container six columns',
                ),
                dcc.ConfirmDialog(
                        id='stop_confirm',
                        message='the corresponding extension size exceed our limitation (1G)',
                    ),
                html.Div([
                    html.Div(
                        [
                    html.Span("Semantics:", style={"font-weight": "bold",'marginRight': '1%','marginLeft': '0.5%'}),


                    dcc.Dropdown(
                    id="check_semantics",
                    options=[
                        {'label': 'Preferred and Stable', 'value': 'preferred_stable'},
                        {'label': 'Stable and Stage', 'value': 'stable_stage'},
                        #{'label': 'Stable and Stage2', 'value': 'stage2_stable'},
                        {'label': 'Stable and CF2', 'value': 'stable_cf2'},
                        #{'label': 'Stage2 and CF2', 'value': 'cf2_stage2'},
                        #{'label':"Semi-Stable and Preferred", 'value':'semi-stable_preferred'},
                        {'label':"Others", 'value':'others'}
                    ],
                    #value=['preferred_stable'],
                    placeholder="Select semantics pair",
                    style={'height': '30px', 'width': '200px'}
                    ) ,
                    dcc.Store(id='memory-semantic'),
                    #     ],
                    # style={"width": "17%"},
                    # ),
                    html.Div(
                        [
                    # dcc.Checklist(
                    #     id="check_semantics2",
                    #     options=[
                    #         {'label': 'Preferred ', 'value': 'preferred'},
                    #         {'label': 'Stable', 'value': 'stable'},
                    #         {'label': 'Stage', 'value': 'stage'},
                    #         {'label': 'Stage2', 'value': 'stage2'},
                    #         {'label': 'CF2', 'value': 'cf2'},
                    #         {'label': "Semi-Stable", 'value': 'semi-stable'}
                    #     ],
                    #     value=[],
                    #     labelStyle={'display': 'inline-block'},
                    #    # style={'width': '17%'},
                    #
                    # ),

                    dcc.Dropdown(
                        id="check_semantics2",
                        options=[
                            {'label': 'Preferred ', 'value': 'preferred'},
                            {'label': 'Stable', 'value': 'stable'},
                            {'label': 'Stage', 'value': 'stage'},
                            #{'label': 'Stage2', 'value': 'stage2'},
                            {'label': 'CF2', 'value': 'cf2'},
                            {'label': "Semi-Stable", 'value': 'semi-stable'}
                        ],
                        placeholder="Select Semantics",
                        #value=['MTL', 'NYC'],
                        multi=True,
                        style={'height': '30px', 'width': '200px'}
                    )

                        ],
                    id="check_semantics2_style",

                    style={"display": "none"},
                    ),
                    dcc.Checklist(
                        id="default_params",
                        options=[{"label": "use suggested parameters", "value": "use_bo"}],
                        className="dcc_control",
                        value=[],
                        style={'height': '50px', 'marginLeft': '1.5%'}
                    ),
                    # html.Abbr("\u003F", title="Hello, I am hover-enabled helpful information.")
                    html.Span(
                        "?",
                        id="tooltip-target",
                        style={
                            "textAlign": "center",
                            "color": "white"
                        },
                        className="dot"),

                    dbc.Tooltip(
                        "use Bayesian Optimization to find suitable clustering parameters (Eps, MinPts, Cluster Num)",
                        target="tooltip-target",)
                        ],
                className="row flex-display"
                ),
                dbc.Tooltip(
                    "you can choose one or two semantics",
                    target="check_semantics2_style"),

                dcc.Store(id="store-prev-comparisons"),
                html.Br(),
                html.Div([
                    html.P("Eps:", style={"font-weight": "bold"}, className="dcc_control"),
                    dcc.Input(id='eps', type="number", placeholder="input eps",disabled=False, style={'width': '22%','marginRight': '4%'}),
                    dbc.Tooltip("DBscan  parameter, specifies the distance between two points to be considered within one cluster.suggested a decimal in range[1,3]", target="eps"),
                    html.P("MinPts:", style={"font-weight": "bold"}, className="dcc_control"),
                    dcc.Input(id='minpts',  type="number", placeholder="input minpts",disabled=False,style={'width': '22%'}),#style={'width': '10%','marginRight': '0.5%'}
                    dbc.Tooltip("DBscan parameter, the minimum number of points to form a cluster. suggested an integer in range[3,15]", target="minpts"),
                ],

                className="row flex-display"),
                html.Br(),
                html.Div([
                    html.P("Cluster Num", style={"font-weight": "bold"}, className="dcc_control"),
                    dcc.Input(id='cluster_num', type="number", placeholder="input cluster num",disabled=False),
                    dbc.Tooltip("Kmeans parameter, number of clusters, suggested an integer in range[2,15]", target="cluster_num"),
                ],
                className="row flex-display"),
                html.Button(id='submit-button-state', n_clicks=0, children='Submit',className="middle-button",style={'font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"})],
                    className="empty_container five columns",

                #className="pretty_container five columns",) style={'width': '32%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}
                )
            ],
            className="row flex-display"
            ),

            dcc.Tab(label='PROCESSED', children=[

                html.Div([
                            html.P("Derived Extensions"),
                            html.Ul(id="extension-list", children=get_file_name(EXTENSION_DIR),style={'height':350,'overflowX':'scroll'}),
                            #html.Pre(id='hover-data', children=get_file_name(EXTENSION_DIR),style=styles['pre'])
                        ],
                            className="pretty_container four columns"),
                html.Div([
                            html.P("Processed Documents"),
                            html.Ul(id="processed-list", children=get_file_name(ZIP_DIRECTORY),style={'height':350,'overflowX':'scroll'})
                        ],
                            className="pretty_container four columns"),



                html.Div(
                [
                    # html.Div(
                    #     [
                    html.A(
                        html.Button("Get Default Data", id="get_default_data",className="middle-button",
                                    style={"color":"#FFFF","backgroundColor":"#2F8FD2"}),
                        href="https://github.com/Lexise/ASP-Analysis/tree/master/data/default_raw_data",
                    ),
                    #     ],
                    #     #className="one-fifth column",
                    #     style={'width': '15%', 'textAlign': 'left'},
                    #     id="download_default",
                    # ),
                    dcc.Upload([html.Button("Upload Processed Data" ,id="view_button",className="middle-button")],id="view-processed-button",multiple=True),

                    dcc.ConfirmDialog(
                        id='confirm',
                        message='Do you want to visualize uploaded data?',
                    ),
                    html.Div(id='hidden-div', style={'display':'none'})
                    #html.A('Refresh', href='/')
                ],
                className="one-fifth column",
                #style={'width': '15%', 'textAlign': 'right'},
                id="button",
                ),
             ],
             className="row flex-display"       ),


        ]),

        html.Div([
            html.Br(),
            dbc.Progress(
                [
                    dbc.Progress(id="progress-extension", children="extension computing", value=20, color="success",
                                 bar=True, animated=False),
                    dbc.Progress(id='progress-process', children="data processing", value=30, color="warning",
                                 bar=True, animated=False),
                    dbc.Progress(value=20, color="danger", bar=True, animated=False),
                ],
                multi=True,
            )],

        )

        ],
        id="upload_block",
        #style={'display':'none'},
        className="my_container"
        ),


        #html.Br(),
        # html.Div([
        #     html.Div([
        #
        #         html.Div([
        #             html.P("Upload",className="dcc_control",style={"width":"91%",'textAlign': 'left'}),
        #             html.Button(id='clear-upload', n_clicks=0, children='Clear',style={'width': '12%','font-size':'11px','textAlign': 'right'}),
        #
        #         ],
        #             className="row flex-display",
        #             #style={'textAlign': 'right'}
        #         ),
        #         html.Ul(id="file-list", children=get_file_name(UPLOAD_DIRECTORY))
        #     ],
        #         className="pretty_container seven columns"),
        #
        #     html.Div([
        #         html.P("Processed"),
        #         html.Ul(id="processed-list", children=get_file_name(ZIP_DIRECTORY))
        #     ],
        #         className="pretty_container seven columns")
        # ],
        #     className="row flex-display"
        # )

        ],
    className='my_main_container',
    #style={'height': '100%'}
)







ThreeD_analysis=html.Div([
                dcc.Link(html.Button('back'), href='/'),
                # html.Div([
                #     dcc.Graph(id="3d_scatter_cluster",className="row flex-display"),
                #     dcc.Graph(id="3d_scatter_group", className="row flex-display"),
                #     ],
                #     className="row flex-display"
                # ),
                    html.Div([

                        html.Div([
                                    html.P("Dimensional Reduction Method:", style={"font-weight": "bold"},className="dcc_control"),
                                    dcc.RadioItems(
                                        id="reduction_method",
                                        options=[
                                            {"label": "Tsne ", "value": "tsne"},
                                            {"label": "SVD", "value": "svd"},
                                            {"label": "AutoEncode", "value": "auto"},
                                        ],
                                        labelStyle={"display": "inline-block"},
                                        value="tsne",
                                        className="dcc_control",
                                )],
                                style={'marginRight': '2%', 'width': '30%'},
                                className="row flex-display"
                        ),
                        html.Div([
                                    html.P("Cluster Algorithm:", style={"font-weight": "bold"}, className="dcc_control"),
                                    dcc.RadioItems(
                                        id="clustering-method",
                                        options=[
                                            {"label": "DBscan ", "value": "db"},
                                            {"label": "Kmeans", "value": "km"},
                                        ],
                                        labelStyle={"display": "inline-block"},
                                        value="db",
                                        className="dcc_control",
                                    )],

                            style={ 'width': '30%'},
                            className="row flex-display"
                            )

                        ],
                        className = "row flex-display"),
                    ],
                )

app.layout = html.Div([


    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    # hidden signal value
    #html.Div(id='signal', style={'display': 'none'}),
    dcc.Loading(
        id="loading-2",
        type="default",
        children=html.Div(id="signal")
    ),

],
    #className='my_main_container',
     style={'height':'100vh', 'margin':dict(l=0, r=0, b=0, t=0)}
    #style={"display": "flex", "flex-direction": "column"},
)



def global_individual(eps, minpts, n_cluster, use_optim, semantics):
    files = uploaded_files(UPLOAD_DIRECTORY)
    start_time0 = time.process_time()
    question = ""
    try:
        apx_files = []
        extensions=[]
        for x in files:
            if x.endswith('.apx'):
                # Assume that the user uploaded a CSV file
                question = x
                apx_files.append(x)
            else:
                extensions.append(x)


    except Exception as e:
        print(e)
        return html.Div([
            'There was no input file.'
        ])
        # zipname=files[0].strip("apx")+"zip"
    try:
       print('cleaned:', clean_folder(PROCESSED_DIRECTORY))
       temp_para=[]
       #process_individual_semantics = Process_data("start")
       for a_semantic in semantics:

          if len(extensions) == 0:
              extension=process_data.process_extension_individual(question, a_semantic, PROCESSED_DIRECTORY, UPLOAD_DIRECTORY,
                                           EXTENSION_DIR, ASP_DIR)


          else:
              extension = process_data.find_semantic_files(extensions, a_semantic)

          transfered, arguments, itemlist = process_data.initial_process_individual(PROCESSED_DIRECTORY,
                                                                           UPLOAD_DIRECTORY + question,
                                                                           EXTENSION_DIR + extension, a_semantic)
          temp_para.append({'semantic':a_semantic,'transfered':transfered,'arguments':arguments,'itemlist':itemlist})

       process_data.get_catogery(PROCESSED_DIRECTORY, semantics)
       for item in temp_para:
            process_data.process_data_two_sets(PROCESSED_DIRECTORY, UPLOAD_DIRECTORY + question, item['transfered'], item['arguments'], item['itemlist'], eps,
                             minpts,
                             n_cluster, use_optim, item['semantic'])
       process_data.addional_process_individual(PROCESSED_DIRECTORY,semantics)
       print("whole process time consuming: ", time.process_time() - start_time0)
    except Exception as e:
        print('error:',e)
        return html.Div([
            'input data is not proper.'
        ])


    return True

def global_store(eps, minpts, n_cluster, use_optim,semantics):
    # simulate expensive query
    files = uploaded_files(UPLOAD_DIRECTORY)
    question=""

    start_time0 = time.process_time()
    try:
        apx_files=[]
        extensions = []
        for x in files:
            if x.endswith('.apx'):
                # Assume that the user uploaded a CSV file
                question = x
                apx_files.append(x)
            else:
                extensions.append(x)


    except Exception as e:
        print(e)
        return html.Div([
            'There was no input file.'
        ])
        #zipname=files[0].strip("apx")+"zip"
    try:
        if len(extensions) == 0:
            extension_dir=EXTENSION_DIR
            end=""
            # if 'cf2_stage2' in semantics:
            #
            #
            #     return global_individual(eps, minpts, n_cluster,use_optim, ["cf2","stage2"])

            if "preferred_stable" in semantics:
                asp_encoding="prefex.dl"
                end="PR"
            elif "stable_stage" in semantics:
                asp_encoding="stage-cond-disj.dl"
                end="STG"
            # elif "stage2_stable" in semantics:
            #     asp_encoding="stage2_gringo_versus_stable.lp"
            #     end="STG2"
            elif "stable_cf2" in semantics:
                asp_encoding="cf2_gringo_versus_stable.lp"
                end="CF2"
            # elif "semi-stable_preferred" in semantics:
            #     pass
            extension_file="{}.EE_{}".format(question, end)
            #manager = Manager()
            # string = []
            # P = Process(target=compute_extensions, args=(UPLOAD_DIRECTORY +question,asp_encoding,EXTENSION_DIR+extension_file,string))
            #
            # P.start()
            compute_extension_result=compute_extensions(UPLOAD_DIRECTORY +question,ASP_DIR+asp_encoding,EXTENSION_DIR+extension_file)
            #print("done\n : string is ",string)
            if compute_extension_result=='oversize': #len(string) >0 and string[0]=='oversize':
                return 'oversize'
        else:
            extension_dir=UPLOAD_DIRECTORY
            extension_file = extensions[0]
            compute_extension_result='finished' #extension already be calculated, no need to do it again

    except Exception as e:
        print('error:',e)
        return html.Div([
            'input data is not proper.'
        ])
    if question!="" :
        print("finish extensions computing:", time.process_time() - start_time0)
        start_time = time.process_time()#time.time()
        print("start process")
        #process_data=Process_data('start')
        result=process_data.process_data(PROCESSED_DIRECTORY,UPLOAD_DIRECTORY+question, extension_dir+extension_file,eps, minpts, n_cluster,use_optim,semantics)
        if not result:
            return html.Div([
                'no extensions exist for the selected semantics'
            ])
        elif result=='parameter_mistakes':
            return html.Div([
                'parameter mistakes'
            ])
        #process_data(PROCESSED_DIRECTORY, UPLOAD_DIRECTORY + question, UPLOAD_DIRECTORY + stg_answer, eps, minpts, n_cluster)

        print("(whole)get processed data", time.process_time() - start_time) #time.time() - start_time)
        if compute_extension_result == 'finished':
            return result
    else:
        print("the form of input file is not correct.")
        return False

# @app.callback(
#     Output("popover", "is_open"),
#     [Input("eps", "clickData")],
#     [State("popover", "is_open")],
# )
# def toggle_popover(n, is_open):
#     if n:
#         return not is_open
#     return is_open
@app.callback(Output('check_semantics2_style', 'style') ,[Input("check_semantics", "value")])
def show_other_option(semantics):
    if semantics=="others":
        return {'display': 'block', 'width': '17%','marginLeft': '0.5%'}

    return {'display': 'none'}


# @app.callback(Output('clustering-method', 'options') ,[Input("memory-semantic", "data")])
# def show_other_option(semantics):
#     if len(semantics)<2:
#         return dash.exceptions.PreventUpdate
#     else:
#      return semantics


@app.callback([Output('signal', 'children'),Output('memory-semantic', 'data'),] ,[Input('submit-button-state', 'n_clicks'),Input("check_semantics", "value")],
              [State("default_params", "value"),State('eps', 'value'), State('minpts', 'value'), State('cluster_num', 'value'),State('store-prev-comparisons', 'data')])
def compute_value( n_clicks, semantics, use_optim, eps, minpts, n_cluster,semantics2):
    # compute value and send a signal when done

    if len(os.listdir(UPLOAD_DIRECTORY)) == 0:
        print("return no content")
        return "", None   #haven't upload data
    else:
            if semantics=="others":
                if semantics2 is None:
                    raise dash.exceptions.PreventUpdate
                return global_individual(eps, minpts, n_cluster,use_optim, semantics2), semantics2
            if int(n_clicks)>0:
                # if len(os.listdir(PROCESSED_DIRECTORY)) != 0:
                #     clean_folder(PROCESSED_DIRECTORY)

                return [global_store(eps, minpts, n_cluster,use_optim, semantics),semantics]
            return "",semantics
       #already  process, no need to pass data again
    # global_store(value)
    # return value




@app.callback([Output('store-prev-comparisons', 'data')],
                     [Input('check_semantics', 'value'),Input('check_semantics2', 'value'),Input('submit-button-state', 'n_clicks')])
def select_comparison(semantic1,comparisons, submit_click):


  if semantic1== None or submit_click==0:
      raise dash.exceptions.PreventUpdate
  if semantic1!='others' or comparisons is None:  # on page load
      return dash.no_update

  if len(comparisons) >2 or len(comparisons) == 2:
    # changes store-prev-comparisons which triggers above callback
    return comparisons[0:2],

  elif len(comparisons) ==1:
      print("comparison:",comparisons)
      return comparisons

  else:
    # when <= 3 don't modify store-prev-comparisons and therefore don't trigger above
    raise dash.exceptions.PreventUpdate#dash.no_update




@app.callback(
    Output("file-list", "children"),
    [Input("clear-upload","n_clicks"), Input('signal', 'children'),Input('upload-data', 'filename'),Input('upload-ex-data', 'filename')],
    [State('upload-data', 'contents'),State('upload-ex-data', 'contents')]
)
def update_output(clear_click,children,uploaded_filenames,uploaded_ex_filenames, uploaded_file_contents,uploaded_ex_file_contents ):
    """Save uploaded files and regenerate the file list."""
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if len(os.listdir(UPLOAD_DIRECTORY)) != 0 and 'clear-upload' in changed_id:#and n_click==None:
            clean_folder(UPLOAD_DIRECTORY)
            return ""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
            for name, data in zip(uploaded_filenames, uploaded_file_contents):#[],[]
                save_file(name, data, UPLOAD_DIRECTORY)
                #save_file(name, data, EXTENSION_DIR)

    if uploaded_ex_filenames is not None and uploaded_ex_file_contents is not None:
            for name, data in zip(uploaded_ex_filenames, uploaded_ex_file_contents):#[],[]
                save_file(name, data, UPLOAD_DIRECTORY)

    files = uploaded_files(UPLOAD_DIRECTORY)
    if len(files) == 0:
        return ""
    else:
        FILE_LIST=[html.Li(file_download_link(filename)) for filename in files]
        return FILE_LIST

# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("bar_chart", "figure")],
)


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-argument':
        return argument_analysis
    elif pathname == '/page-correlation':
        return correlation_page
    elif pathname=="/page-3d":
        return ThreeD_analysis
    else:
        return main_page


@app.callback(
    Output('3d_scatter_group', 'hoverData'),
    [Input('3d_scatter_cluster', 'hoverData')]) #Input('scatter_cluster', 'hoverData'),
def connect_hover(hover_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if hover_data:
        if "3d_scatter_cluster" in changed_id:
            print("hover data:", hover_data)
            #selected_id=hover_data['points'][0]['customdata']

            selected_point_info=hover_data["points"][0]
            print("selected id:", selected_point_info)
            if len(selected_point_info)==1:
                return None
            selected_point_id=selected_point_info["customdata"]
            return hover_data

@app.callback(
    Output('hover-data', 'children'),
    [Input("visual-dropdown","value"),Input('scatter_groups', 'hoverData'),Input("clustering-method", "value"),Input('confirm', 'submit_n_clicks')]) #Input('scatter_cluster', 'hoverData'),
def display_click_data(drop_selection, clickData2, method,n_click):
    if drop_selection and drop_selection=='upload-user':
        return None
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if len(os.listdir(PROCESSED_DIRECTORY)) == 12: # or len(os.listdir(PROCESSED_DIRECTORY)) == 5:
           processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "CombinedProcessed_data.pkl")
           label="category"
    else:
        label="groups"
        if  len(os.listdir(PROCESSED_DIRECTORY))==6 or n_click:#load and processed

                processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")
        else:
                processed_data = pd.read_pickle(DEFAULT_DATA + 'processed_data.pkl')



    cluster_label = method + "_cluster_label"

    print("changed_id:",[p['prop_id'] for p in dash.callback_context.triggered])
    # if clickData1 or clickData2:
    #     if "scatter_cluster" in changed_id:
    #         clickData=clickData1
    #         print("clickData1",clickData1)
    #     elif "scatter_groups" in changed_id:
    #         clickData = clickData2
    #     else:
    #         return None
    if clickData2:
        if "scatter_groups" in changed_id:
            print("clickData2", clickData2)
            selected_point_info=clickData2["points"][0]
            if len(selected_point_info)==1:
                return None
            selected_point_id=selected_point_info["customdata"]

            selected_point=processed_data[processed_data.id==selected_point_id]
            selected_point_arg=list(selected_point.arg.tolist()[0])
            selected_point_arg=','.join(x for x in selected_point_arg)
            print("argument set:",selected_point_arg)
            selected_point_cluster=selected_point[cluster_label]
            return html.Div([
                    html.Table([
                        html.Tr([html.Th('Id'),
                                 html.Th('Cluster'),
                                 html.Th('Semantics Label'),
                                 html.Th('Arguments'),
                                 # html.Th('Most Recent Click')
                                 ]),
                                 html.Tr([html.Td(selected_point_id),
                                          html.Td(selected_point_cluster),
                                          html.Td(selected_point[label]),
                                          html.Td(selected_point_arg),
                                          # html.Td(button_id)
                                          ])
                                 ],
                    )
                    ],
            className = "pretty_container")
    return None



@app.callback([Output("semantic-method-1", "options"), Output("semantic-method","style")], [Input("store-prev-comparisons", "data"),Input('signal', 'children') ])
def chnage_selection(semantic_checks, signal_content):
    print("current semantics: {}".format(semantic_checks))

    if len(os.listdir(PROCESSED_DIRECTORY)) == 6:
        return [],{"display": "none"}
    elif len(os.listdir(PROCESSED_DIRECTORY)) == 12:
        if semantic_checks is not None:
            if len(semantic_checks) ==2:
                return [{"label": semantic_checks[0], "value": semantic_checks[0]}],{"display": "block"}
            else:
                return [], {"display": "none"}
        else:
            semantics=get_current_processed_dir_semantic(PROCESSED_DIRECTORY)

            return [{"label": semantics[0], "value": semantics[0]}, {"label":semantics[1], "value": semantics[1]}],{'marginLeft': '2%', 'width': '18%'}


    return [],{"display": "none"}

@app.callback(Output('scatter_cluster', 'figure'),
                [ Input('signal', 'children'),Input("dimensional-reduction1", "value"),
                Input("clustering-method", "value"),  Input('confirm', 'submit_n_clicks'), Input('memory-semantic', 'data'), Input("semantic-method-1",'value')])

def generate_tabs1( content, reduction1,  method, n_click,semantic, present_semantic):#processed_data, table1_data,table2_data ):
    try:
        if len(os.listdir(PROCESSED_DIRECTORY)) == 12: #or semantic == "cf2_stage2":

            processed_data = pd.read_pickle(PROCESSED_DIRECTORY + present_semantic+"_processed_data.pkl")

        else:

            if  len(os.listdir(PROCESSED_DIRECTORY))==6 or n_click:#load and processed
                   processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")
            else:
                processed_data = pd.read_pickle(DEFAULT_DATA + 'processed_data.pkl')


        if reduction1=="svd":
            x_axe="svd_position_x"
            y_axe="svd_position_y"
        elif reduction1=="tsne":
            x_axe = "tsne_position_x"
            y_axe = "tsne_position_y"
        else:
            x_axe = "auto_position_x"
            y_axe = "auto_position_y"

        cluster_label = method + "_cluster_label"

        inputdata=processed_data.copy()
        inputdata[cluster_label]=["Cluster "+str(a) for a in processed_data[cluster_label]]
        cluster_set = list(processed_data[cluster_label].unique())
        if len(cluster_set)<=52:
            temp = [100 if x == -1 else None for x in cluster_set]

            clusters_symbol=temp
        else:
            clusters_symbol=[None]*len(cluster_set)

        #let hover on the plot directly
        # figure1 = px.scatter(inputdata, x=x_axe, y=y_axe,  color=cluster_label, symbol=clusters_symbol,#symbol_sequence =[102]*len(inputdata),
        #                      hover_name=cluster_label,  hover_data={
        #                                                              x_axe:False, # remove species from hover data
        #                                                               y_axe: False,
        #                                                               cluster_label:False,
        #                                                              'id':True,
        #                                                              'arg':True, # add other column, default formatting
        #                                                              'groups':True, # add other column, customized formatting
        #                                                             }
        #                      )
        # #figure1.update_traces() #remove hover  hovertemplate=None,
        # figure1.update_xaxes(showgrid=False,visible=False,zerolinecolor="Black")
        # figure1.update_yaxes(showgrid=False,zeroline=True,visible=False,zerolinecolor="black")
        # figure1.update_layout(  clickmode='event+select', plot_bgcolor='rgba(0,0,0,0)',legend_title_text='',autosize=True)
        # figure1 = go.Figure(figure1)

        #remove hover, provide an extra table to show the arg information of selected data point

        fig = go.Figure()
        for x in cluster_set:
            fig.add_trace(go.Scatter(
                x=processed_data[processed_data[cluster_label]==x][x_axe],
                y=processed_data[processed_data[cluster_label]==x][y_axe],
                customdata=processed_data[processed_data[cluster_label]==x]["id"],
                mode='markers',
                name=str(x)+" cluster",
                #hovertemplate='Id:%{customdata} ',
                # hovertext=processed_data[processed_data[cluster_label]==x].arg,
                hoverinfo="none",#'none'
                marker=dict(
                    symbol=clusters_symbol[x]
                ),
                showlegend=True
            ))
        fig.update_layout(xaxis={'showgrid': False,'visible': False,},
                          yaxis={'showgrid': False, 'visible': False, },
                          plot_bgcolor='rgba(0,0,0,0)',
                          clickmode='event+select')
        return fig
    except Exception as e:
        print(e)
        raise dash.exceptions.PreventUpdate
        #return dash.no_update
    # radio_item=[
    #             html.Span("Semantics:", style={"margin-top": "5%","font-weight": "bold"}),
    #             dcc.RadioItems(
    #                 id="semantic-method-1",
    #                 options=[
    #                     {"label": "CF2", "value": "cf2"},
    #                     {"label": "Stage2", "value": "stg2"},
    #                 ],
    #                 labelStyle={"display": "inline-block"},
    #                 value="cf2",
    #             )]


@app.callback([Output('scatter_groups', 'figure'),Output('scatter_groups', 'style'),
               Output('3d_scatter_cluster', 'figure'),Output('3d_scatter_group', 'figure'),Output('3d_scatter_cluster', 'style'),Output('3d_scatter_group', 'style'),
               Output('upload_block','style'), Output('options-visualization','style')],
              [Input('signal', 'children'),
               Input('visual-dropdown', 'value'),
               Input("semantic-method-1", "value"),
               Input("dimensional-reduction1", "value"),
               Input("clustering-method", "value"),
               Input('confirm', 'submit_n_clicks')]
               #Input('memory-semantic', 'data')],

              )
def generate_tabs(content, selected_visual, semantic, reduction2, cluster_method, n_click):
            if selected_visual=='dropdown-2d':
                fig=generate_tabs2(content,reduction2, cluster_method, n_click)
                return fig,{'display':'block'},{},{},{'display':'none'},{'display':'none'},{'display':'none'},{'display':'flex','position': 'relative' }
            elif selected_visual=='dropdown-3d':
                fig1,fig2=display3d(reduction2, semantic, cluster_method)
                return {},{'display':'none'},fig1,fig2,{'display':'block'},{'display':'block'},{'display':'none'},{'display':'flex','position': 'relative' }
            elif selected_visual=='upload-user':
                return {},{'display':'none'},{}, {},{'display':'none'},{'display':'none'},{'display':'block'},{'display':'none'}
            else:
                raise dash.exceptions.PreventUpdate#return {},{'display':'none'},{}, {},{'display':'none'},{'display':'none'}



# @cache.memoize(TIMEOUT)
def generate_tabs2(content, reduction2, method, n_click):  # method):

    if len(os.listdir(PROCESSED_DIRECTORY)) == 12 and os.path.isfile(PROCESSED_DIRECTORY + "CombinedProcessed_data.pkl"): #or semantic =="cf2_stage2":

        color_label="category"
        processed_data=pd.read_pickle(PROCESSED_DIRECTORY + "CombinedProcessed_data.pkl")#pd.concat([present_data1,present_data2,present_common])


    else:
        color_label = "groups"
        if len(os.listdir(PROCESSED_DIRECTORY)) == 6 or n_click:  # load and processed
                processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")

        else:
                processed_data = pd.read_pickle(DEFAULT_DATA + 'processed_data.pkl')

    if reduction2 == "svd":
        x_axe = "svd_position_x"
        y_axe = "svd_position_y"
    else:
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"
    #cluster_label = method + "_cluster_label"


        # let hover on the plot directly
        # figure2 = px.scatter(processed_data, x=x_axe, y=y_axe, color="groups",
        #                      hover_name="groups", hover_data={x_axe: False,  # remove species from hover data
        #                                                       y_axe: False,
        #                                                       cluster_label: True,
        #                                                       'id': True,
        #                                                       'arg': True,  # add other column, default formatting
        #                                                       'groups': False,  # add other column, customized formatting
        #
        #                                                       })
        # figure2.update_xaxes(showgrid=False, visible=False)
        # figure2.update_yaxes(showgrid=False, zeroline=True, visible=False, zerolinecolor="black")
        # figure2.update_layout(plot_bgcolor='rgba(0,0,0,0)', legend_title_text='Groups', autosize=True)
        # figure2 = go.Figure(figure2)

        # remove hover, provide an extra table to show the arg information of selected data point

    fig = go.Figure()

    # add shape (groups)
    groups_set = np.sort(processed_data[color_label].unique())
    get_color_label(processed_data, color_label, groups_set)

    for x in groups_set:
        fig.add_trace(go.Scatter(
            x=processed_data[processed_data[color_label] == x][x_axe],
            y=processed_data[processed_data[color_label] == x][y_axe],
            customdata=processed_data[processed_data[color_label] == x]["id"],
            mode='markers',
            name=str(x),
            hoverinfo="none",
            marker=dict(
                size=14,
                color=processed_data[processed_data[color_label] == x]['color'],
                # colorscale='Viridis',
                # opacity=0.8
            ),
            # marker=dict(
            #     color=processed_data[processed_data[color_label] == x]['color'],
            #
            # ),
            showlegend=True
        ))

    # Add points
    cluster_label = method + "_cluster_label"
    #processed_data[cluster_label] = ["Cluster " + str(a) for a in processed_data[cluster_label]]
    cluster_set = list(processed_data[cluster_label].unique())
#    colors=get_colors(len(cluster_set))#['blue','red','green','yellow']
    num_color=len(cluster_set)
    if num_color>8:
        colors=get_colors(len(cluster_set))#['rgb'+str(x) for x in sns.color_palette(n_colors=len(cluster_set))]
    else:
        colors=[WELL_COLOR_new[i] for i in range(num_color)]
    # for clabel in cluster_set:
    #     select=processed_data[processed_data[cluster_label] == clabel]
    #     x_list = select[x_axe]
    #     y_list = select[y_axe]
    #     # fig.add_shape(type="circle",
    #     #               xref="x", yref="y",
    #     #               x0=x_list.min(), y0=y_list.min(),
    #     #               x1=x_list.max(), y1=y_list.max(),
    #     #               opacity=0.3,
    #     #               layer="below",
    #     #               fillcolor=colors[cluster_set.index(clabel)],
    #     #               line_color=colors[cluster_set.index(clabel)],
    #     #               name='cluster'+str(clabel)
    #     #               )
    data=processed_data
    for cls in cluster_set:
        fig.add_trace(
            go.Scatter(
                x=data[data[cluster_label] == cls][x_axe],
                y=data[data[cluster_label] == cls][y_axe],
                #z=[len(set(s)) for s in data[data[cluster_label] == cls]["arg"]],
                customdata=data[data[cluster_label]==cls]['id'],

                text=data[data[cluster_label] == cls].id,
                # text=["cluster: {}".format(x) for x in data[data[cluster_label]==cls][cluster_label]],
                mode='markers',
                name="cluster" + str(cls + 1),
                marker=dict(

                    color=cls,  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.8,

                ),
            ))



        # go.Scatter(
        #     x=x_list,
        #     y=y_list,
        #     #fill="toself",
        #     mode='markers',
        #
        #     # marker=dict(
        #     #                   size=15,
        #     #                   color=colors[cluster_set.index(clabel)],
        #     #     #colorscale='Viridis',
        #     #     opacity=0.06
        #     # ),
        #     marker=dict(
        #         #size=4,
        #         #color=colors[cluster_set.index(clabel)],
        #         #
        #         colorscale='Viridis',
        #         opacity=0.8
        #     ),
        #     customdata=select["id"],
        #     name=clabel,
        #     #line_color=colors[cluster_set.index(clabel)],
        #     #text=clabel,
        #
        # )


    #fig.layout.plot_bgcolor = '#386cb0'

    fig.update_layout(xaxis={'showgrid': False, 'visible': False, },
                      yaxis={'showgrid': False, 'visible': False, },
                      plot_bgcolor='rgba(0,0,0,0)',
                      clickmode='event+select',  # autosize=True

                      width=1500,
                      height=700,
                      autosize=False,
                      margin={'t':0,'l':0,'r':0,'b':0},

                      )



    return fig#, {'display': 'block'}



# @app.callback([ Output('table1', 'children'),
#                Output('table2', 'children')],
#               [Input('signal', 'children'),
#                Input("clustering-method", "value"),
#                Input('confirm', 'submit_n_clicks'),Input("semantic-method-1", "value"),])
# # @cache.memoize(TIMEOUT)
# def generate_tabs3(content,   cluster_method,
#                    n_click, table_method):  # processed_data, table1_data,table2_data ):
#     if len(os.listdir(PROCESSED_DIRECTORY)) == 12:
#         group_table = pd.read_pickle(PROCESSED_DIRECTORY + "group_feature.pkl")
#         # semantics=get_current_processed_dir_semantic(PROCESSED_DIRECTORY)
#         # if table_method not in semantics:
#         #     cluster_table = pd.read_pickle(
#         #         PROCESSED_DIRECTORY + semantics[0] + "_" + cluster_method + "_cluster_feature.pkl")
#         # else:
#         #     cluster_table = pd.read_pickle(PROCESSED_DIRECTORY + table_method+ "_"+cluster_method+ "_cluster_feature.pkl")
#         cluster_table = pd.read_pickle(
#             PROCESSED_DIRECTORY + table_method + "_" + cluster_method + "_cluster_feature.pkl")
#     elif len(os.listdir(PROCESSED_DIRECTORY)) == 6 or n_click:  # load and processed
#
#             group_table = pd.read_pickle(PROCESSED_DIRECTORY + "group_feature.pkl")
#             cluster_table = pd.read_pickle(PROCESSED_DIRECTORY +  cluster_method + "_cluster_feature.pkl")
#
#
#     else:
#             group_table = pd.read_pickle(DEFAULT_DATA+"group_feature.pkl")
#             if cluster_method == "km":
#                 cluster_table = pd.read_pickle(DEFAULT_DATA+"km_cluster_feature.pkl")
#             else:
#                 cluster_table = pd.read_pickle(DEFAULT_DATA+"db_cluster_feature.pkl")
#
#
#     # table
#     if len(group_table) == 0:
#         table1 = html.H5("No group feature")
#     else:
#         table1 = dash_table.DataTable(
#             data=group_table.to_dict('records'),
#             columns=[{"name": i, "id": i} for i in group_table.columns],
#             style_table={
#                 'maxHeight': '300px',
#                 'overflowY': 'scroll'
#             },
#             style_header={
#                 'fontWeight': 'bold'
#             },
#             style_cell={
#                 'font_size': '20px',
#                 'text_align': 'center'
#             },
#         )
#
#     if not len(cluster_table):
#         table2 = html.H5("No cluster Feature")
#     else:
#         table2 = dash_table.DataTable(
#             data=cluster_table.to_dict('records'),
#             columns=[{"name": i, "id": i} for i in cluster_table.columns],
#
#             style_table={
#                 'maxHeight': '300px',
#                 'overflowY': 'scroll'
#             },
#             style_header={
#                 'fontWeight': 'bold'
#             },
#             style_cell={
#                 'font_size': '20px',
#                 'text_align': 'center'
#             },
#         )
#     return  table1, table2


@app.callback(
    [Output("bar_chart", "figure"),Output("my-range-slider","figure")],
    [Input("data_present_selector", "value"),Input("my-range-slider", "value"),Input("sort_selector", "value"),Input("semantic-method-1-1", "value")],
    )
@cache.memoize(TIMEOUT)
def make_bar_figure(present_data, valuelist,sort_state,semantics):
    if os.path.isfile(PROCESSED_DIRECTORY +"CombinedBar_data.pkl") and semantics=="combined":
        dataset_bar = pd.read_pickle(PROCESSED_DIRECTORY + "CombinedBar_data.pkl")
    elif len(os.listdir(PROCESSED_DIRECTORY)) == 12 and os.path.isfile(PROCESSED_DIRECTORY + "CombinedProcessed_data.pkl"): #change it later

        dataset_bar = pd.read_pickle(PROCESSED_DIRECTORY + semantics+"_bar_data.pkl")
        #processed_data=pd.read_pickle(PROCESSED_DIRECTORY + "CombinedProcessed_data.pkl")#pd.concat([present_data1,present_data2,present_common])


    else:
        color_label = "groups"
        if len(os.listdir(PROCESSED_DIRECTORY)) == 6:  # load and processed
                dataset_bar = pd.read_pickle(PROCESSED_DIRECTORY + "bar_data.pkl")

        else:
                dataset_bar = pd.read_pickle(DEFAULT_DATA+'bar_data.pkl')



    slider=dict(
        min = 0,
        max = len(dataset_bar),
        step = 1,
        value = [1, int(0.5*len(dataset_bar))]
    )
    if present_data == "all":
       if sort_state == ["decreased"]:
           temp=dataset_bar.sort_values(by=['frequency'],ascending=False, inplace=False)
           figure= set_bar_figure(temp, valuelist)
       else:
           figure= set_bar_figure(dataset_bar, valuelist)
    else:
       dataset=dataset_bar[~dataset_bar.rate.isin([0,100])]
       min = int(valuelist[0] * len(dataset) / len(dataset_bar))
       max= int(valuelist[1] * len(dataset) / len(dataset_bar))
       if sort_state == ["decreased"]:
           temp=dataset.sort_values(by=['frequency'],ascending=False, inplace=False)
           figure = set_bar_figure(temp, [min,max])
       else:
           figure = set_bar_figure(dataset, [min,max])
    return figure,slider


def set_bar_figure(argument_data, valuelist):

    select_idx=range(valuelist[0],valuelist[1])
    selected=argument_data.iloc[select_idx]
    selected["order"]=range(len(selected))

    data = [dict(
            type="bar",
            x=list(selected["argument"]),
            y=list(selected["rate"]),
            hovertext={"fontsize":20},
            #hovertext=["attribute:{arg},rate:{percent}".format(arg=row.attribute,percent=row.rate) for index,row in selected.iterrows()],
            name="All Wells"
        )]

    layout_count={}

    layout_count["title"] = "Rate/Argument"

    layout_count["dragmode"] = "select"
    layout_count["showlegend"] = False
    layout_count["autosize"] = True,
    layout_count["titlefont"] = {"size": 25}
    layout_count["marker"] = {"fontsize": 10}
    if 'xaxis' in layout_count:
        del layout_count['xaxis']
        del layout_count['yaxis']
    figure = dict(data=data, layout=layout_count)
    return figure

# @app.callback(
#     [Output("confirm", "displayed")],
#     [Input("view-processed-button", "filename"), Input("view-processed-button", "contents"),Input("view_button","n_clicks")]
#     )
# def run_processed_data(uploaded_filenames, uploaded_file_contents,n_click ):
#     if n_click is None:
#         return [False]
#     if len(os.listdir(PROCESSED_DIRECTORY))!=0 and uploaded_filenames is not None and n_click%6==1:
#         clean_folder(PROCESSED_DIRECTORY)
#     print("click number: ",n_click)
#     print("upload_name:",uploaded_filenames)
#     if uploaded_filenames is not None and uploaded_file_contents is not None:
#         for name, data in zip([uploaded_filenames], [uploaded_file_contents]):
#             save_file(name, data, PROCESSED_DIRECTORY)
#         if len(os.listdir(PROCESSED_DIRECTORY)) >=6:
#             return [True]
#     return [False]

#
# @app.callback([Output("confirm", "displayed")],
#               [Input('view-processed-button', 'contents')],
#               [State('view-processed-button', 'filename')])
# def update_output(content, name):
#     if content is None:
#         return [False]
#     #for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
#         # the content needs to be split. It contains the type and the real content
#     content_type, content_string = content.split(',')
#     # Decode the base64 string
#     content_decoded = base64.b64decode(content_string)
#     # Use BytesIO to handle the decoded content
#     zip_str = io.BytesIO(content_decoded)
#     # Now you can use ZipFile to take the BytesIO output
#     zip_obj = ZipFile(zip_str, 'r')
#     zip_obj.extractall(PROCESSED_DIRECTORY)
#     return [True]

@app.callback([Output("stop_confirm", "displayed"),Output("progress-extension", "animated"),Output("progress-process", "animated")] ,  #should be style
       [Input('submit-button-state', 'n_clicks'), ], #Input("signal","children"),
        [State('signal', 'children')]
              )
def show_confirm(n_clicks,value):

    if n_clicks>0:
        if value=='oversize':
            return True, False, False
        elif value=='True':
            return False, False, True
        else:
            return False, True, False

    else:
        return False, False, False

@app.callback(Output("confirm", "displayed"),
       [Input("hidden-div","figure")])
def show_confirm(value):
    if value :
        #print(n_click)
        if len(os.listdir(PROCESSED_DIRECTORY))==6:
            print("test:", value)
            return True
    return False


@app.callback(
    [Output("eps", "disabled"),Output("minpts", "disabled"),Output('cluster_num', "disabled")],
    [Input("default_params", "value")],
    )
@cache.memoize(TIMEOUT)
def make_bar_figure(defaut_params):
    print("default param",defaut_params)
    if not defaut_params:
        return False,False,False #for now
    else:
        return True,True,True


@app.callback(Output('hidden-div', 'figure'),
              [Input('view-processed-button', 'contents'),Input("view_button","n_clicks")],
              [State('view-processed-button', 'filename')])
def update_output(contents, n_click, name):
    if contents is None:
        return None
    #for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        # the content needs to be split. It contains the type and the real content
    for content in contents:
        content_type, content_string = content.split(',')
        # Decode the base64 string
        content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
        zip_str = io.BytesIO(content_decoded)
        # Now you can use ZipFile to take the BytesIO output
        zip_obj = ZipFile(zip_str, 'r')
        zip_obj.extractall(PROCESSED_DIRECTORY)

    if  len(os.listdir(PROCESSED_DIRECTORY))==6: #n_click>=2 and n_click%2==0 and
        files = zip_obj.namelist()
        return files
    else:
        return None

@app.callback([Output('processed-list', 'children'),Output("extension-list",'children'),],
              [ Input('signal', 'children'),Input('hidden-div', 'figure'), Input('confirm', 'submit_n_clicks')])#

def update_output(children1, children2, n_click):
    if  children2:
        return [html.Li(file_download_link(filename)) for filename in children2],get_file_name(EXTENSION_DIR)

    return  get_file_name(ZIP_DIRECTORY),get_file_name(EXTENSION_DIR)



@app.callback(
    [Output("selected_cluster","children"),
        Output('info-container', 'children')
        ,Output("pie_graph", "figure")],
    [Input('bar_chart', 'clickData') ,Input("clustering-method","value"), Input("semantic-method-1-1", "value")]) #Input("check_semantics","value")


def update_cluster_rate(clickData, cluster_method, semantics):
    if len(os.listdir(PROCESSED_DIRECTORY)) == 12:  # load and processed

        process_data = pd.read_pickle(PROCESSED_DIRECTORY + semantics + "_processed_data.pkl")
    elif len(os.listdir(PROCESSED_DIRECTORY)) == 6:
        process_data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")
    else:

            process_data =  pd.read_pickle(DEFAULT_DATA + 'processed_data.pkl')

    mini_block=[]
    layout_pie={}
    layout_pie["title"] = "Cluster Summary"
    if clickData is None:
        return "Selected Argument: None",mini_block,dict(data=None, layout=layout_pie),
    temp=clickData["points"][0]
    arguments=re.search(r'\d+', temp["x"]).group()  #int()
    selected=[]
    result0= "Selected Argument:{}  \n".format(arguments)
    for index, row in process_data.iterrows():
        if arguments in row.arg:
            selected.append(index)
    if len(selected) == 0:
        return "No data has this argument",mini_block,dict(data=None, layout=layout_pie),
    data = process_data.loc[selected]
    result=""
    cluster_label=cluster_method+"_cluster_label"
    clusters=set(data[cluster_label])
    for cluster in clusters:
        num=len(data[data[cluster_label]==cluster])
        result=result+"{} % belong to cluster {} . ".format(num/len(data)*100,cluster)
    semantics=data["groups"].unique()

    for semantic in semantics:
        percent_value = len(data[data.groups == semantic]) / len(data) * 100
        precent = "{:.2f}".format(percent_value) + "%"
        current=html.Div(
            [html.H6(precent), html.P(semantic)],
            className="mini_container",
            )
        mini_block.append(current)
                        # semantic_first=semantics[0]
                        # semantic_second=semantics[1]
                        # stable_value=len(data[data.groups == semantic_first])/ len(data) * 100
                        # stable = "{:.2f}".format(stable_value) + "%"
                        # prefer_value = len(data[data.groups == semantic_second]) / len(data) * 100
                        # other = "{:.2f}".format(prefer_value) + "%"
    # if "preferred" in semantics:
    #     prefer_value=len(data[data.groups == "preferred-"])/ len(data) * 100
    #     other="{:.2f}".format(prefer_value)+"%"
    #     pr_display={'display':'block'}
    #     stg_display={'display':'none'}
    # else:
    #     stage_value = len(data[data.groups == "stage"]) / len(data) * 100
    #     other = "{:.2f}".format(stage_value) + "%"
    #     pr_display = {'display': 'none'}
    #     stg_display = {'display': 'block'}

    # x = [html.Div(
    #     [html.H6(stable), html.P(semantic_first)],
    #     id="stable_block",
    #     className="mini_container",
    # ),
    #     html.Div(
    #         [html.H6(other), html.P(semantic_second)],
    #         id="prefer_block",
    #         className="mini_container",
    #
    #     )]


    result = dict({
        "cluster": [],
        "num": []
    })
    for cluster in clusters:
        result["cluster"].append(str(cluster) + " cluster")
        num = len(data[data[cluster_label] == cluster])
        result["num"].append(num)
        # result["rate"].append(num/len(data))
    if len(clusters) > 8:
        r = lambda: random.randint(0, 255)
        data_bar = [
            dict(
                type="pie",
                labels=result["cluster"],
                values=result["num"],
                name="Production Breakdown",
                text=[
                    "Data Num in cluster {}".format(a) for a in result["cluster"]
                ],
                hoverinfo="text+value+percent",
                textinfo='none',
                hole=0.5,
                marker=dict(colors=['#%02X%02X%02X' % (r(), r(), r()) for i in clusters])

            )
        ]
    else:
        data_bar = [
            dict(
                type="pie",
                labels=result["cluster"],
                values=result["num"],
                name="Production Breakdown",
                text=[
                    "Data Num in cluster {}".format(a) for a in result["cluster"]
                ],
                hoverinfo="text+value+percent",
                textinfo="label+percent+name",
                hole=0.5,
                marker=dict(colors=[WELL_COLOR_new[i] for i in clusters]),

            )
        ]


    layout_pie["legend"] = dict(
        font=dict(color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)"
    )

    pie_figure = dict(data=data_bar, layout=layout_pie)

    result2=""
    for group in set(data.groups):
        num = len(data[data.groups == group])
        result2 = result2 + "{} % belong to group {}. ".format(num / len(data) * 100, group)


    return result0, mini_block, pie_figure


@app.callback(
    Output('basic-interactions', 'figure'),
    [Input('bar_chart', 'clickData'), Input("argument-dimensional-reduction","value"), Input("clustering-method","value"),Input("semantic-method-1-1", "value")])

def update_graph(clickData, dimensional_reduction, cluster_method,semantics):
    if len(os.listdir(PROCESSED_DIRECTORY)) == 12:  # load and processed

        process_data = pd.read_pickle(PROCESSED_DIRECTORY + semantics + "_processed_data.pkl")
    elif len(os.listdir(PROCESSED_DIRECTORY)) == 6:
        process_data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")
    else:

        process_data = pd.read_pickle(DEFAULT_DATA + 'processed_data.pkl')


    layout_scatter ={"title":"Distribution of Selected Argument",
                       "clickmode":'event+select'
                       }


    if clickData is None:
        return {"layout":layout_scatter}
    temp=clickData["points"][0]
    cluster_label=cluster_method +"_cluster_label"
    arguments=re.search(r'\d+', temp["x"]).group()#int()
    selected=[]
    for index, row in process_data.iterrows():
        if arguments in row.arg:
            selected.append(index)
    data=process_data.loc[selected]
    unselected_data=process_data[~process_data.index.isin(selected)]
    if dimensional_reduction=="svd":
        x_axe="svd_position_x"
        y_axe="svd_position_y"
    elif dimensional_reduction=="tsne":
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"
    else:
        x_axe = "auto_position_x"
        y_axe = "auto_position_y"
    return {
        'data': [
            dict(
            x=data[x_axe],
            y=data[y_axe],
            text=["clusters: {}".format(x) for x in data[cluster_label]],
            name="selected",
            mode='markers',
            marker={
                'size': 12,
                'opacity': 1.0,
                'line': {'width': 0.5, 'color': 'white'}
            }),
            dict(
                x = unselected_data[x_axe],
                y = unselected_data[y_axe],
                text=["clusters: {}".format(x) for x in unselected_data[cluster_label]],
                name = "unselected",
                mode = 'markers',
                marker= { 'size': 12,
                "color":"LightSkyBlue",
                'opacity': 0.3,
                'line': {'width': 0.5, 'color': 'white'}
                # make text transparent when not selected
                #'textfont': {'color': 'rgba(0, 0, 0, 0)'}
            }
            )
        ],
        'layout': layout_scatter
    }

# @app.callback([Output('3d_scatter_cluster', 'figure'),Output('3d_scatter_group', 'figure'),Output('3d_scatter_cluster', 'style'),Output('3d_scatter_group', 'style')],
#             [ Input('dropdown-3d', 'n_clicks'),Input('dropdown-2d', 'n_clicks'), Input("dimensional-reduction1","value"), Input("clustering-method","value"),]
#               )
def display3d( reduction_method, semantics,cluster_method):
    #print("3d click:", selected_3d)
    if len(os.listdir(PROCESSED_DIRECTORY)) == 12:  # or semantic == "cf2_stage2":
        try:
            data = pd.read_pickle(PROCESSED_DIRECTORY + "CombinedProcessed_data.pkl")
            group_label='category'
        except IOError:
            print("File not accessible")


    else:
        group_label = 'groups'
        if len(os.listdir(PROCESSED_DIRECTORY)) == 6:  # load and processed
            data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")
        else:
            data = pd.read_pickle(DEFAULT_DATA + 'processed_data.pkl')

    # if os.listdir(PROCESSED_DIRECTORY)==5:  # load and processed
    #
    #         data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")
    #
    # else:
    #
    #         data = pd.read_pickle(DEFAULT_DATA + "processed_data.pkl")

    if reduction_method=="svd":
        x_axe="svd_position_x"
        y_axe="svd_position_y"
    elif reduction_method=="tsne":
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"
    else:
        x_axe = "auto_position_x"
        y_axe = "auto_position_y"
    cluster_label = cluster_method + "_cluster_label"
    cluster_set = data[cluster_label].unique()
    print('cluster selected:', cluster_label)
    print('cluster label:', cluster_set)
    # fig = go.Figure(go.Scatter3d(
    #     x=data[x_axe],
    #     y=data[y_axe],
    #     z=[len(set(s)) for s in data["arg"]],
    #     text=["clusters: {}".format(x) for x in data[cluster_label]],
    #     name=data[cluster_label].tolist(),
    #     mode='markers',
    #     marker=dict(
    #         size=12,
    #         color=data[cluster_label],  # set color to an array/list of desired values
    #         colorscale='Viridis',  # choose a colorscale
    #         opacity=0.8
    #     )
    # ),
    #
    #
    # )
    #processed_data[processed_data[cluster_label] == cls][x_axe]
    text_list=[]
    inputdata=data.copy()
    for index, row in inputdata.iterrows():
        arg_list=list(row.arg)
        inserted_arg=[x for y in (arg_list[i:i + 6] + ["<br>"] * (i < len(arg_list) - 5) for
                     i in range(0, len(arg_list), 6)) for x in y]
        one_arg = str(inserted_arg).strip('[]')
        #input_arg=f_comma(one_arg, group=20, char='<br>')
        input_arg=one_arg.strip('"') #remove ' from string
        text_list.append(input_arg)

    inputdata["arguments"]=text_list
    symbols=['circle', 'square', 'cross', 'square-open', 'x','diamond', 'diamond-open', 'circle-open']
    if len(cluster_set)<=len(symbols):
        cluster_symbol=symbols
    else:
        cluster_symbol=[None]*len(cluster_set)
    print('cluster_symbol:', cluster_symbol)
    fig = go.Figure(data=[go.Scatter3d(
            x=data[data[cluster_label]==cls][x_axe],
            y=data[data[cluster_label]==cls][y_axe],
            z=[len(set(s)) for s in data[data[cluster_label]==cls]["arg"]],
            #customdata=data[data[cluster_label]==cls]['id'],
            customdata=['<br>'+s for s in inputdata[inputdata[cluster_label]==cls]["arguments"]],
            hovertemplate='Id:%{text}</b><br>length:%{z} <br></b>Arguments:%{customdata} ',
            text=data[data[cluster_label]==cls].id,
            #text=["cluster: {}".format(x) for x in data[data[cluster_label]==cls][cluster_label]],
            mode='markers',
            name="cluster"+str(cls+1),
            marker=dict(
                size=3,
                color=cls,  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8,
                symbol=cluster_symbol[cls-1]
            ),

        ) for cls in cluster_set
        ],
        layout=go.Layout(title=dict(
            text='Clusters Distribution',
            xref="paper",
            yref="paper",
            #x=0.5,
            font=dict(
                size=20,
            )
        ),

            autosize=False,
            width=650,
            height=700,
            showlegend=True,
        )
    )

    group_set = data[group_label].unique()
    fig2 = go.Figure(data=[go.Scatter3d(
        x=data[data[group_label] == cls][x_axe],
        y=data[data[group_label] == cls][y_axe],
        z=[len(set(s)) for s in data[data[group_label] == cls]["arg"]],
        # text=["groups: {}".format(x) for x in data[data[group_label] == cls][group_label]],

        customdata=['<br>' + s for s in inputdata[inputdata[group_label] == cls]["arguments"]],
        hovertemplate='Id:%{text}</b><br>length:%{z} <br></b>Arguments:%{customdata} ',
        text=data[data[group_label] == cls].id,
        mode='markers',
        name= str(cls),
        marker=dict(
            size=3,
            #color=cls,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    ) for cls in group_set
    ],
    layout=go.Layout( title=dict(
                                text='Semantics Distribution',
                                xref="paper",
                                yref="paper",
                                #x=0.5,
                                font=dict(
                                    size=20,
                                )
                        ),
                        autosize=False,
                        width=650,#700,
                        height=700,
                        showlegend=True,
    ))
    # fig2.update_layout(
    #
    #     autosize=False,
    #     width=850,
    #     height=850,
    # )
    return fig,fig2#,{'display':'block'},{'display':'block'}


@app.callback(Output('correlation_hm', 'figure'),
              [Input('btn-nclicks-1', 'n_clicks'),Input('btn-nclicks-2', 'n_clicks'),Input('btn-nclicks-3', 'n_clicks'),Input('btn-nclicks-4', 'n_clicks')])
def displayClick(btn1, btn2 , btn3, btn4):

    if  len(os.listdir(PROCESSED_DIRECTORY)) == 6:  # load and processed

            data_correlation = pd.read_pickle(PROCESSED_DIRECTORY + "correlation_matrix.pkl")
            processed_data=pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")

    else:

            data_correlation = pd.read_pickle(DEFAULT_DATA + "correlation_matrix.pkl")
            processed_data=pd.read_pickle(DEFAULT_DATA + "processed_data.pkl")

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]


    round_correlation = data_correlation.copy()
    threshold=1/(2*len(processed_data))
    for idx, raw in round_correlation.iterrows():
        for x in raw.index:
            to_round_value=raw[x]
            if -threshold<=to_round_value<=threshold:
                round_correlation.loc[idx, x]=0
            else:
                round_correlation.loc[idx, x] = round(to_round_value, 10)

    abs_correlation = round_correlation.copy() #data_correlation
    for idx, raw in abs_correlation.iterrows():
        for x in raw.index:
            abs_correlation.loc[idx, x] = abs(raw[x])

    if btn1%2:
        temp_round_correlation=round_correlation.copy()
        distances = np.sqrt((1 - abs_correlation) / 2)
        res_order = compute_serial_matrix(distances.values, method='single')
        new_order = [abs_correlation.index[i] for i in res_order]

        ordered_correlation_matrix = round_correlation.reindex(index=new_order, columns=new_order) #data_correlation.reindex
        z_value=ordered_correlation_matrix.to_numpy()
        # original_z = z_value.copy()
        # a = pd.DataFrame(data=original_z, index=new_order, columns=new_order)
        # a.to_pickle("method1.pkl")
        #z_value[z_value==0]=np.nan
        x_value=[str(x)+"arg" for x in new_order]
        y_value=[str(x)+"arg" for x in new_order]

    elif btn2%2:
        all_new_order=innovative_correlation_clustering(round_correlation)
        new_test = round_correlation.reindex(index=all_new_order, columns=all_new_order)#data_correlation.reindex
        z_value=new_test.to_numpy()
        # original_z = z_value.copy()
        # a = pd.DataFrame(data=original_z, index=all_new_order, columns=all_new_order)
        # a.to_pickle("method2.pkl")
        #z_value[z_value == 0] = np.nan
        x_value=[str(x) + "arg" for x in new_test.columns]
        y_value=[str(x) + "arg" for x in new_test.index]

    elif btn3 % 2:
        new_order=abs_optimal_leaf_ordering(data_correlation)
        ordered_correlation_matrix = round_correlation.reindex(index=new_order, columns=new_order)#data_correlation.reindex
        z_value = ordered_correlation_matrix.to_numpy()
        # original_z = z_value.copy()
        # a = pd.DataFrame(data=original_z, index=new_order, columns=new_order)
        # a.to_pickle("method3.pkl")
        #z_value[z_value == 0] = np.nan
        x_value = [str(x) + "arg" for x in new_order]
        y_value = [str(x) + "arg" for x in new_order]
    elif btn4 % 2:
        all_new_order=my_optimal_leaf_ordering(round_correlation)
        new_test = round_correlation.reindex(index=all_new_order, columns=all_new_order)#data_correlation.reindex
        z_value = new_test.to_numpy()
        # original_z = z_value.copy()
        # a=pd.DataFrame(data=original_z, index=all_new_order, columns=all_new_order)
        # a.to_pickle("method4.pkl")
        #z_value[z_value == 0] = np.nan
        x_value = [str(x) + "arg" for x in new_test.columns]
        y_value = [str(x) + "arg" for x in new_test.index]
    else:
        z_value=round_correlation.to_numpy()#data_correlation.reindex
        #original_z=z_value.copy()
        #z_value[z_value == 0] = np.nan
        x_value=[str(x)+"arg" for x in round_correlation.columns]
        y_value=[str(x)+"arg" for x in round_correlation.index]



    fig = go.Figure(go.Heatmap(
        z=z_value,
        x=x_value,
        y=y_value,
        #customdata =original_z,
        hovertemplate='value:%{z} <br><b>x:%{x}</b><br>y: %{y} ',
        name='',
        colorscale='RdBu',
       ))
    fig.update_layout(
        autosize=False,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            autorange='reversed'
        ),
        height=750)
    fig.update_xaxes(side="top")
    return fig



@app.callback([ Output('feature_semantic_table', 'children'),
               Output('feature_cluster_table', 'children')],
              [Input('signal', 'children'),
               Input("clustering-method", "value"),
               Input('confirm', 'submit_n_clicks'),Input("semantic-method-1", "value"),])
def get_feature_table(content,   cluster_method,
                   n_click, table_method):
    if len(os.listdir(PROCESSED_DIRECTORY)) == 12:
        group_table = pd.read_pickle(PROCESSED_DIRECTORY + "group_feature.pkl")
        cluster_table = pd.read_pickle(
            PROCESSED_DIRECTORY + table_method + "_" + cluster_method + "_cluster_feature.pkl")
    elif len(os.listdir(PROCESSED_DIRECTORY)) == 6 or n_click:  # load and processed

        group_table = pd.read_pickle(PROCESSED_DIRECTORY + "group_feature.pkl")
        cluster_table = pd.read_pickle(PROCESSED_DIRECTORY + cluster_method + "_cluster_feature.pkl")


    else:
        group_table = pd.read_pickle(DEFAULT_DATA + "group_feature.pkl")
        if cluster_method == "km":
            cluster_table = pd.read_pickle(DEFAULT_DATA + "km_cluster_feature.pkl")
        else:
            cluster_table = pd.read_pickle(DEFAULT_DATA + "db_cluster_feature.pkl")

        # table
    if len(group_table) == 0:
        table1 = "No Semantics Didentifier"
    else:
        table1=dbc.Table.from_dataframe(group_table, striped=True, bordered=True, hover=True)


    if len(cluster_table) == 0:
        table2 = "No Cluster Didentifier"
    else:
        table2=dbc.Table.from_dataframe(cluster_table, striped=True, bordered=True, hover=True)

    return [table1, table2]


if __name__ == '__main__':
    app.run_server(debug=True)