import dash
import re
import time
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
import plotly.graph_objects as go
from flask_caching import Cache
from flask import send_from_directory, request
from clustering_correlation import compute_serial_matrix,innovative_correlation_clustering,my_optimal_leaf_ordering,abs_optimal_leaf_ordering
import numpy as np
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
if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)
if not os.path.exists(CACHE_DIRECTORY):
    os.makedirs(CACHE_DIRECTORY)
if not os.path.exists(ZIP_DIRECTORY):
    os.makedirs(ZIP_DIRECTORY)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.scripts.config.serve_locally=True
app.css.config.serve_locally=True
server = app.server
cache_config = {
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR":CACHE_DIRECTORY,
}

# Empty cache directory before running the app
clean_folder(CACHE_DIRECTORY)

# Resources 
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

@server.route("/data")
def data():
    reduction = request.args['reduction'] if 'reduction' in request.args else 'tsne' #tsne, svd, auto
    cluster_method = request.args['cluster_method'] if 'cluster_method' in request.args else 'db' #db, km
    data, color_label = get_data()

    x_axe = reduction + "_position_x"
    y_axe = reduction + "_position_y"
    method = cluster_method + '_cluster_label'
    data['z'] = [len(set(s)) for s in data["arg"]]
    
    return data[['id', x_axe, y_axe, 'z', method, color_label]].rename(
        columns = {
            x_axe :'x',
            y_axe :'y',
            method : 'c',
            color_label: 'gc'
        }).T.to_json()

@server.route("/data_args")
def data_args():
    id = request.args['id'] # required
    data, _ = get_data()

    data = data[ [str(x) == str(id) for x in data['id']] ]
    return data[['id', 'arg']].T.to_json()


def get_data(n_click = None, name = "processed_data", check_combined = True):
    if check_combined and len(os.listdir(PROCESSED_DIRECTORY)) == 12 and os.path.isfile(PROCESSED_DIRECTORY + "CombinedProcessed_data.pkl"): #or semantic =="cf2_stage2":
        color_label = "category"
        processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "CombinedProcessed_data.pkl") #pd.concat([present_data1,present_data2,present_common])
    else:
        color_label = "groups"
        if len(os.listdir(PROCESSED_DIRECTORY)) == 6 or n_click:  # load and processed
            processed_data = pd.read_pickle(PROCESSED_DIRECTORY + name + ".pkl")
        else:
            processed_data = pd.read_pickle(DEFAULT_DATA + name + '.pkl')
        
    return processed_data, color_label

@server.route("/arguments")
def arguments(): 
    dataset_bar, _ = get_data(name = "bar_data")
    dataset_bar['argument'] = dataset_bar['argument'].apply(lambda x : x.replace('argument', '')) #not sure if this is needed for uploaded data

    return dataset_bar.rename(
        columns = {
            'argument' :'a',
            'frequency' :'f',
            'rate' : 'r',
        }).T.to_json()

@server.route("/correlation")
def correlation(): 
    matrix = request.args['matrix'] if 'matrix' in request.args else '' #hrp, r-hrp, olo, r-olo
    x, y, z = get_correlation(matrix)
    data = pd.DataFrame({'x': x, 'y': y, 'z': [e for e in z] })
    data['x'] = data['x'].apply(lambda x : x.replace('arg', '')) # check if this "arg" can be removed before this
    data['y'] = data['y'].apply(lambda x : x.replace('arg', ''))
    
    return data.T.to_json()

app.config.suppress_callback_exceptions = True
# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("bar_chart", "figure")],
)

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

if len(os.listdir(PROCESSED_DIRECTORY))==12:
    processed_semantics=get_current_processed_dir_semantic(PROCESSED_DIRECTORY)
else:
    processed_semantics=['stable','preferred']
dataset_all=pd.read_pickle(DEFAULT_DATA+'bar_data.pkl')

cache = Cache()
cache.init_app(app.server, config=cache_config)

TIMEOUT = 120

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
    html.Div([
        html.Div([
            html.P("Presented data:", style={"fontWeight": "bold"},className="control_label"),
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

            html.P("Cluster Algorithm:", style={"fontWeight": "bold"}, className="control_label"),
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
            html.Div(
                id="info-container",
                className="row container-display"
            )],
            className="pretty_container four columns",
            id="cross-filter-options",
        ),
        html.Div([
            dcc.Graph(id="bar_chart"),
            dcc.RangeSlider(
                id='my-range-slider',
                min=0,
                max=len(dataset_all),
                step=1,
                value=[int(0.2 * len(dataset_all)), int(0.5 * len(dataset_all))]
            ),],
            className="pretty_container seven columns",
            style={'width': '64%'}
        ),],
        className="row flex-display",
    ),
    html.Div([
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="basic-interactions"),
        )],
        className="row",
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='basic-interactions'),
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
            ),],
            className="pretty_container seven columns"
        ),

        html.Div(
            [dcc.Graph(id="pie_graph")],
            className="pretty_container five columns",
        ),],
        className = "row flex-display",
    ),],
)

correlation_page= html.Div([
    dcc.Link(html.Button('back'), href='/'),
    html.Div([
        dcc.Graph(id="correlation_hm"),
        html.Div([
            html.Button('HRP',
                        style={'fontSize': '14px','marginLeft': '2%', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                        id='btn-nclicks-1', n_clicks=0),
            html.Button('Revised HRP',
                        style={'fontSize': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                        id='btn-nclicks-2', n_clicks=0),
            html.Button('OLO',
                        style={'fontSize': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                        id='btn-nclicks-3', n_clicks=0),
            html.Button('Revised OLO',
                        style={'fontSize': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                        id='btn-nclicks-4', n_clicks=0),],
            className="row flex-display"
        )],
        className="pretty_container"
    )]
)

main_page = html.Div([
    # empty Div to trigger javascript file for graph resizing
    html.Div(id="output-clientside"),
    html.Div([
        dcc.Dropdown(
            id='visual-dropdown',
            options=[
                {'label': 'new', 'value': 'upload-user'},
                {'label': '2D', 'value': 'dropdown-2d'},
                {'label': '3D', 'value': 'dropdown-3d'},
            ],
            value='upload-user',
            style={'fontSize': '14px', 'width': '150px','marginLeft': '2%'}
        ),

        dcc.Link(
            html.Button('Argument Analysis', style={'marginLeft': '4%','marginRight': '8px','width': '300px', 'fontSize': '14px', "color": "#FFFF", "backgroundColor": "#2F8FD2"}),
            href='/page-argument'
        ),

        dcc.Link(
            html.Button('Correlation Matrix', style={'marginLeft': '4%', 'width': '300px', 'fontSize': '14px', "color": "#FFFF","backgroundColor": "#2F8FD2"}), 
            href='/page-correlation'
        ),],
        className="row flex-display",
    ),

    html.Div([
        html.Div([ dcc.Graph( id="scatter_groups",style={'display':'none'}) ], className="bare_container"),

        html.Div([
            dcc.Graph(id="3d_scatter_cluster",className="row flex-display",style={'display':'none'}),
            dcc.Graph(id="3d_scatter_group", className="row flex-display",style={'display':'none'}),
            ],

            className="row flex-display",
            style={
                'borderRadius': '5px',
                'margin': '12px',
                'padding': '17px'
            }
        ),], className="row flex-display",
    ),

    html.Div([
        html.Div([
            html.Span("Semantics:", style={"marginTop": "5%","fontWeight": "bold"}),
            dcc.RadioItems(
                id="semantic-method-1",
                options=[
                    {"label": processed_semantics[0], "value": processed_semantics[0]},
                    {"label": processed_semantics[1], "value": processed_semantics[1]},
                ],
                labelStyle={"display": "inline-block"},
                value=processed_semantics[0],
            )],
            id="semantic-method",
            style={'marginLeft': '2%', 'width': '18%'},
        ),
        html.Div(children=[
            html.Span("Dimensional Reduction:", style={"fontWeight": "bold"}),
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
            html.Span("Cluster Algorithm:", style={"fontWeight": "bold"}),
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
                    style={'marginTop':'0.5%','fontSize': '14px', "color": "#000000", "backgroundColor": "#f0f0f0"}),
        dbc.Tooltip(

            id='feature_semantic_table',
            target="feature_semantic",
            style ={'fontSize': '11px',"color": "#000000", "backgroundColor": "#f0f0f0"}
        ),
        html.Button(id='feature_cluster', children='clusters identifier',className='middle-button',
                    style={'marginTop':'0.5%','fontSize': '14px','marginLeft': '3%', "color": "#000000", "backgroundColor": "#f0f0f0"}),#'marginTop':'1%',

        dbc.Tooltip(
            id='feature_cluster_table',
            target="feature_cluster",
            style={'fontSize': '11px', "color": "#000000", "backgroundColor": "#f0f0f0"}
        ),



        ],
            id='options-visualization',
            style={'display':'none'},
            className="row flex-display"
    ),

    html.Div(id='hover-data'),
    html.Div([
        dcc.Tabs([

            dcc.Tab(label='UPLOAD', children=[
                html.Div([
                    html.Div([
                        dcc.Upload([html.Button(children='UPLOAD APX', id="upload_button")], id="upload-data", multiple=True),
                        dcc.Upload([html.Button(children='UPLOAD Extensions', id="upload_ex_button")], id="upload-ex-data", multiple=True),
                    ],
                    className = "row flex-display"),
                    html.Div([
                        html.Div([
                            html.P("Uploaded",className="dcc_control",style={"width":"80%",'textAlign': 'left'}),
                            html.Button(id='clear-upload', n_clicks=0, children='Clear',style={'fontSize':'11px','textAlign': 'left','marginRight': '1%'}),],
                            className = "row flex-display"
                        ),

                        html.Ul(
                            id="file-list", 
                            children=get_file_name(UPLOAD_DIRECTORY))],
                            className="pretty_container",
                    ),],
                    className ='empty_container six columns',
                ),
                dcc.ConfirmDialog(
                    id='stop_confirm',
                    message='the corresponding extension size exceed our limitation (1G)',
                ),
                html.Div([
                    html.Div([
                        html.Span("Semantics:", style={"fontWeight": "bold",'marginRight': '1%','marginLeft': '0.5%'}),
                        dcc.Dropdown(
                            id="check_semantics",
                            options=[
                                {'label': 'Preferred and Stable', 'value': 'preferred_stable'},
                                {'label': 'Stable and Stage', 'value': 'stable_stage'},
                                {'label': 'Stable and CF2', 'value': 'stable_cf2'},
                                {'label':"Others", 'value':'others'}
                            ],
                            placeholder="Select semantics pair",
                            style={'height': '30px', 'width': '200px'}
                        ),
                        dcc.Store(id='memory-semantic'),
                        html.Div([
                            dcc.Dropdown(
                                id="check_semantics2",
                                options=[
                                    {'label': 'Preferred ', 'value': 'preferred'},
                                    {'label': 'Stable', 'value': 'stable'},
                                    {'label': 'Stage', 'value': 'stage'},
                                    {'label': 'CF2', 'value': 'cf2'},
                                    {'label': "Semi-Stable", 'value': 'semi-stable'}
                                ],
                                placeholder="Select Semantics",
                                multi=True,
                                style={'height': '30px', 'width': '200px'}
                            )],
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
                        html.Span(
                            "?",
                            id="tooltip-target",
                            style={
                                "textAlign": "center",
                                "color": "white"
                            },
                            className="dot"
                        ),
                        dbc.Tooltip(
                            "use Bayesian Optimization to find suitable clustering parameters (Eps, MinPts, Cluster Num)",
                            target="tooltip-target",
                        )],
                        className="row flex-display"
                    ),
                    dbc.Tooltip(
                        "you can choose one or two semantics",
                        target="check_semantics2_style"
                    ),

                    dcc.Store(id="store-prev-comparisons"),
                    html.Br(),
                    html.Div([
                        html.P("Eps:", style={"fontWeight": "bold"}, className="dcc_control"),
                        dcc.Input(id='eps', type="number", placeholder="input eps",disabled=False, style={'width': '22%','marginRight': '4%'}),
                        dbc.Tooltip("DBscan  parameter, specifies the distance between two points to be considered within one cluster.suggested a decimal in range[1,3]", target="eps"),
                        html.P("MinPts:", style={"fontWeight": "bold"}, className="dcc_control"),
                        dcc.Input(id='minpts',  type="number", placeholder="input minpts",disabled=False,style={'width': '22%'}),#style={'width': '10%','marginRight': '0.5%'}
                        dbc.Tooltip("DBscan parameter, the minimum number of points to form a cluster. suggested an integer in range[3,15]", target="minpts"),],
                        className="row flex-display"
                    ),
                    html.Br(),
                    html.Div([
                        html.P("Cluster Num", style={"fontWeight": "bold"}, className="dcc_control"),
                        dcc.Input(id='cluster_num', type="number", placeholder="input cluster num",disabled=False),
                        dbc.Tooltip("Kmeans parameter, number of clusters, suggested an integer in range[2,15]", target="cluster_num"),],
                        className="row flex-display"
                    ),
                    html.Button(id='submit-button-state', n_clicks=0, children='Submit',className="middle-button",style={'fontSize':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"})],
                    className="empty_container five columns",
                )
            ],
            className="row flex-display"
            ),

            dcc.Tab(label='PROCESSED', children=[

                html.Div([
                    html.P("Derived Extensions"),
                    html.Ul(id="extension-list", children=get_file_name(EXTENSION_DIR),style={'height':350,'overflowX':'scroll'}),],
                    className="pretty_container four columns"
                ),
                html.Div([
                    html.P("Processed Documents"),
                    html.Ul(id="processed-list", children=get_file_name(ZIP_DIRECTORY),style={'height':350,'overflowX':'scroll'})],
                    className="pretty_container four columns"
                ),
                html.Div([
                    html.A(
                        html.Button("Get Default Data", id="get_default_data",className="middle-button",
                                    style={"color":"#FFFF","backgroundColor":"#2F8FD2"}),
                        href="https://github.com/Lexise/ASP-Analysis/tree/master/data/default_raw_data",
                    ),
                    dcc.Upload([html.Button("Upload Processed Data" ,id="view_button",className="middle-button")],id="view-processed-button",multiple=True),

                    dcc.ConfirmDialog(
                        id='confirm',
                        message='Do you want to visualize uploaded data?',
                    ),
                    html.Div(id='hidden-div', style={'display':'none'})                    
                    ],
                    className="one-fifth column",
                    id="button",
                ),],
                className="row flex-display"
            ),]
        ),

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

        )],
        id="upload_block",
        className="my_container"
    ),],
    className='my_main_container',
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    # hidden signal value: html.Div(id='signal', style={'display': 'none'}),
    dcc.Loading(
        id="loading-2",
        type="default",
        children=html.Div(id="signal")
    ),],
    style={'height':'100vh', 'margin':dict(l=0, r=0, b=0, t=0)}
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
        process_individual_semantics = Process_data("start")
        for a_semantic in semantics:

            if len(extensions) == 0:
                extension=process_individual_semantics.process_extension_individual(question, a_semantic, PROCESSED_DIRECTORY, UPLOAD_DIRECTORY, EXTENSION_DIR, ASP_DIR)
            else:
                extension = process_individual_semantics.find_semantic_files(extensions, a_semantic)

            transfered, arguments, itemlist = process_individual_semantics.initial_process_individual(PROCESSED_DIRECTORY,
                                                                           UPLOAD_DIRECTORY + question,
                                                                           EXTENSION_DIR + extension, a_semantic)
            temp_para.append({'semantic':a_semantic,'transfered':transfered,'arguments':arguments,'itemlist':itemlist})

        process_individual_semantics.get_catogery(PROCESSED_DIRECTORY, semantics)
        for item in temp_para:
            process_individual_semantics.process_data_two_sets(PROCESSED_DIRECTORY, UPLOAD_DIRECTORY + question, item['transfered'], item['arguments'], item['itemlist'], eps,
                             minpts,
                             n_cluster, use_optim, item['semantic'])
        process_individual_semantics.addional_process_individual(PROCESSED_DIRECTORY,semantics)
        print("whole process time consuming: ", time.process_time() - start_time0)
    except Exception as e:
        print('error:',e)
        return html.Div([ 'input data is not proper.' ])

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
            print('computing extensions....')
            compute_extension_result=compute_extensions(UPLOAD_DIRECTORY +question,ASP_DIR+asp_encoding,EXTENSION_DIR+extension_file)
            print('extensions computed!')
            #print("done\n : string is ",string)
            if compute_extension_result=='oversize': #len(string) >0 and string[0]=='oversize':
                print('oversize')
                return 'oversize'
        else:
            extension_dir=UPLOAD_DIRECTORY
            extension_file = extensions[0]
            compute_extension_result='finished' #extension already be calculated, no need to do it again

    except Exception as e:
        print('error:',e)
        return html.Div([ 'input data is not proper.' ])
    if question!="" :
        print("finish extensions computing:", time.process_time() - start_time0)
        start_time = time.process_time()#time.time()
        print("start process")
        process_pair_semantics=Process_data('start')
        result=process_pair_semantics.process_data(PROCESSED_DIRECTORY,UPLOAD_DIRECTORY+question, extension_dir+extension_file,eps, minpts, n_cluster,use_optim,semantics)
        if not result:
            return html.Div([ 'no extensions exist for the selected semantics' ])
        elif result=='parameter_mistakes':
            return html.Div([ 'parameter mistakes' ])
        #process_data(PROCESSED_DIRECTORY, UPLOAD_DIRECTORY + question, UPLOAD_DIRECTORY + stg_answer, eps, minpts, n_cluster)

        print("(whole)get processed data", time.process_time() - start_time) #time.time() - start_time)
        if compute_extension_result == 'finished':
            return result
    else:
        print("the form of input file is not correct.")
        return False

@app.callback(Output('check_semantics2_style', 'style') ,[Input("check_semantics", "value")])
def show_other_option(semantics):
    if semantics=="others":
        return {'display': 'block', 'width': '17%','marginLeft': '0.5%'}

    return {'display': 'none'}

@app.callback(
    [Output('signal', 'children'),Output('memory-semantic', 'data'),] ,
    [Input('submit-button-state', 'n_clicks'),Input("check_semantics", "value")],
    [State("default_params", "value"),State('eps', 'value'), State('minpts', 'value'), State('cluster_num', 'value'),State('store-prev-comparisons', 'data')])
def compute_value( n_clicks, semantics, use_optim, eps, minpts, n_cluster,semantics2):
    
    print("compute value\n" + 
        "n_clicks " + str(n_clicks) + "\n " + 
        "semantics " + str(semantics) + "\n " + 
        "use_optim " + str(use_optim) + "\n " + 
        "eps " + str(eps) + "\n " + 
        "minpts " + str(minpts) + "\n " + 
        "n_cluster " + str(n_cluster) + "\n " + 
        "semantics2 " + str(semantics2)
    )

    if len(os.listdir(UPLOAD_DIRECTORY)) == 0:
        print("return no content")
        return "", None   #haven't upload data
    else:
        if semantics=="others":
            if semantics2 is None:
                raise dash.exceptions.PreventUpdate
            return global_individual(eps, minpts, n_cluster,use_optim, semantics2), semantics2
        if int(n_clicks)>0:
            return global_store(eps, minpts, n_cluster,use_optim, semantics), semantics
        return "", semantics #already  process, no need to pass data again

@app.callback(
    [Output('store-prev-comparisons', 'data')],
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
    [State('upload-data', 'contents'),State('upload-ex-data', 'contents')])
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

@app.callback(Output('page-content', 'children'),[Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-argument':
        return argument_analysis
    elif pathname == '/page-correlation':
        return correlation_page
    else:
        return main_page

@app.callback(
    Output('3d_scatter_group', 'hoverData'),
    [Input('3d_scatter_cluster', 'hoverData')])
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
    [Input("visual-dropdown","value"),Input('scatter_groups', 'hoverData'),Input("clustering-method", "value"),Input('confirm', 'submit_n_clicks')])
def display_click_data(drop_selection, clickData2, method,n_click):
    if drop_selection and drop_selection=='upload-user':
        return None
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    processed_data, label = get_data(n_click)
    cluster_label = method + "_cluster_label"

    print("changed_id:",[p['prop_id'] for p in dash.callback_context.triggered])

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
                    html.Tbody([
                        html.Tr([
                            html.Th('Id'),
                            html.Th('Cluster'),
                            html.Th('Semantics Label'),
                            html.Th('Arguments'),
                            # html.Th('Most Recent Click')
                        ]),
                        html.Tr([
                            html.Td(selected_point_id),
                            html.Td(selected_point_cluster),
                            html.Td(selected_point[label]),
                            html.Td(selected_point_arg),
                            # html.Td(button_id)
                            ])
                    ])
                ])
            ], className = "pretty_container")
    return None

@app.callback([Output("semantic-method-1", "options"), Output("semantic-method","style")], [Input("store-prev-comparisons", "data"),Input('signal', 'children') ])
def chnage_selection(semantic_checks, signal_content):
    print("current semantics: {}".format(semantic_checks))

    if len(os.listdir(PROCESSED_DIRECTORY)) == 6:
        return [],{"display": "none"}
    elif len(os.listdir(PROCESSED_DIRECTORY)) == 12:
        if semantic_checks is not None:
            if len(semantic_checks) ==2:
                return [{"label": semantic_checks[0], "value": semantic_checks[0]}, {"label":semantic_checks[1], "value": semantic_checks[1]}],{}
            else:
                return [], {"display": "none"}
        else:
            semantics=get_current_processed_dir_semantic(PROCESSED_DIRECTORY)

            return [{"label": semantics[0], "value": semantics[0]}, {"label":semantics[1], "value": semantics[1]}],{'marginLeft': '2%', 'width': '18%'}


    return [],{"display": "none"}

@app.callback(
    [
        Output('scatter_groups', 'figure'),
        Output('scatter_groups', 'style'), 
        Output('3d_scatter_cluster', 'figure'), 
        Output('3d_scatter_group', 'figure'), 
        Output('3d_scatter_cluster', 'style'), 
        Output('3d_scatter_group', 'style'),
        Output('upload_block','style'), 
        Output('options-visualization','style')
    ], [
        Input('visual-dropdown', 'value'), # TODO: test if [Input('memory-semantic', 'data'), Input("semantic-method-1", "value")] are also needed.
        Input("dimensional-reduction1", "value"),
        Input("clustering-method", "value"),
        Input('confirm', 'submit_n_clicks')
    ])
def generate_tabs(selected_visual, reduction, cluster_method, n_click):
    if selected_visual=='dropdown-2d':
        fig=display2d(reduction, cluster_method, n_click)
        return fig,{'display':'block'},{},{},{'display':'none'},{'display':'none'},{'display':'none'},{'display':'flex','position': 'relative' }
    elif selected_visual=='dropdown-3d':
        fig1,fig2=display3d(reduction, cluster_method)
        return {},{'display':'none'},fig1,fig2,{'display':'block'},{'display':'block'},{'display':'none'},{'display':'flex','position': 'relative' }
    elif selected_visual=='upload-user':
        return {},{'display':'none'},{}, {},{'display':'none'},{'display':'none'},{'display':'block'},{'display':'none'}

def display2d(reduction, method, n_click):
    processed_data, color_label = get_data(n_click)

    x_axe=reduction + "_position_x"
    y_axe=reduction + "_position_y"
        
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
            ),
            showlegend=True
        ))

    # Add points
    cluster_label = method + "_cluster_label"
    # processed_data[cluster_label] = ["Cluster " + str(a) for a in processed_data[cluster_label]]
    cluster_set = list(processed_data[cluster_label].unique())
    # colors=get_colors(len(cluster_set))#['blue','red','green','yellow']
    num_color=len(cluster_set)
    if num_color>8:
        colors=get_colors(len(cluster_set))#['rgb'+str(x) for x in sns.color_palette(n_colors=len(cluster_set))]
    else:
        colors=[WELL_COLOR_new[i] for i in range(num_color)]

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
            )
        )

    fig.update_layout(
        xaxis={'showgrid': False, 'visible': False, },
        yaxis={'showgrid': False, 'visible': False, },
        plot_bgcolor='rgba(0,0,0,0)',
        #clickmode='event+select',  # autosize=True
        width=1500,
        height=700,
        autosize=False,
        margin={'t':0,'l':0,'r':0,'b':0},
    )

    return fig

def display3d(reduction_method, cluster_method):
    data, group_label = get_data()

    x_axe = reduction_method + "_position_x"
    y_axe = reduction_method + "_position_y"

    cluster_label = cluster_method + "_cluster_label"
    cluster_set = data[cluster_label].unique()
    print('cluster selected:', cluster_label)
    print('cluster label:', cluster_set)
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
    fig = go.Figure(
        data=[go.Scatter3d(
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

        ) for cls in cluster_set ],
        layout=go.Layout(
            title=dict(
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
    fig2 = go.Figure(
        data=[go.Scatter3d(
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
        ) for cls in group_set ],
        layout=go.Layout( 
            title=dict(
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
        )
    )
    return fig, fig2

@app.callback(
    [Output("bar_chart", "figure"),Output("my-range-slider","figure")],
    [Input("data_present_selector", "value"),Input("my-range-slider", "value"),Input("sort_selector", "value")])
@cache.memoize(TIMEOUT)
def make_bar_figure(present_data, valuelist,sort_state):

    dataset_bar, _ = get_data(name = "bar_data")

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
            x=list([ x.replace('argument', '') for x in selected["argument"] ]),
            y=list(selected["rate"]),
            hovertext={"fontSize":20},
            #hovertext=["attribute:{arg},rate:{percent}".format(arg=row.attribute,percent=row.rate) for index,row in selected.iterrows()],
            name="All Wells"
        )]

    layout_count={}

    layout_count["title"] = "Rate/Argument"

    layout_count["dragmode"] = "select"
    layout_count["showlegend"] = False
    layout_count["autosize"] = True,
    layout_count["titlefont"] = {"size": 25}
    layout_count["marker"] = {"fontSize": 10}
    if 'xaxis' in layout_count:
        del layout_count['xaxis']
        del layout_count['yaxis']
    figure = dict(data=data, layout=layout_count)
    return figure

@app.callback(
    [Output("stop_confirm", "displayed"),Output("progress-extension", "animated"),Output("progress-process", "animated")] ,  #should be style
    [Input('submit-button-state', 'n_clicks'), ], #Input("signal","children"),
    [State('signal', 'children')])
def show_confirm(n_clicks,value):
    print('show confirm: ' + str(n_clicks) + " " + str(value))
    if n_clicks>0:
        if value=='oversize':
            return True, False, False
        elif value=='True':
            return False, False, True
        else:
            return False, True, False

    else:
        return False, False, False

@app.callback(Output("confirm", "displayed"),[Input("hidden-div","figure")])
def show_confirm(value):
    if value :
        #print(n_click)
        if len(os.listdir(PROCESSED_DIRECTORY))==6:
            print("test:", value)
            return True
    return False

@app.callback(
    [Output("eps", "disabled"),Output("minpts", "disabled"),Output('cluster_num', "disabled")],
    [Input("default_params", "value")],)
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
              [ Input('signal', 'children'),Input('hidden-div', 'figure'), Input('confirm', 'submit_n_clicks')])
def update_output(children1, children2, n_click):
    if  children2:
        return [html.Li(file_download_link(filename)) for filename in children2],get_file_name(EXTENSION_DIR)

    return  get_file_name(ZIP_DIRECTORY),get_file_name(EXTENSION_DIR)

@app.callback(
    [Output("selected_cluster","children"), Output('info-container', 'children'), Output("pie_graph", "figure")],
    [Input('bar_chart', 'clickData'), Input("clustering-method","value")])
def update_cluster_rate(clickData, cluster_method):
    process_data, _ = get_data(check_combined = False)

    mini_block=[]
    layout_pie={}
    layout_pie["title"] = "Cluster Summary"
    if clickData is None:
        return "Selected Argument: None",mini_block,dict(data=None, layout=layout_pie),
    temp=clickData["points"][0]
    arguments=int(re.search(r'\d+', str(temp["x"])).group())
    selected=[]
    result0= "Selected Argument:{}  \n".format(arguments)
    for index, row in process_data.iterrows():
        if str(arguments) in row.arg:
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
    [Input('bar_chart', 'clickData'), Input("argument-dimensional-reduction","value"), Input("clustering-method","value")])
def update_graph(clickData, dimensional_reduction, cluster_method):

    process_data, _ = get_data(check_combined = False)
    layout_scatter ={"title":"Distribution of Selected Argument", "clickmode":'event+select'}

    if clickData is None:
        return {"layout":layout_scatter}
    temp=clickData["points"][0]
    cluster_label=cluster_method +"_cluster_label"
    arguments=int(re.search(r'\d+', str(temp["x"])).group())
    selected=[]
    for index, row in process_data.iterrows():
        if str(arguments) in row.arg:
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

@app.callback(
    Output('correlation_hm', 'figure'),
    [Input('btn-nclicks-1', 'n_clicks'),Input('btn-nclicks-2', 'n_clicks'),Input('btn-nclicks-3', 'n_clicks'),Input('btn-nclicks-4', 'n_clicks')])
def display_correlation_matrix(btn1, btn2 , btn3, btn4):
    matrix_type = "" #these works as a toggle, not button. Bugged if clicked alternating the modes
    if btn1 % 2:
        matrix_type = 'hrp'
    elif btn2 % 2:
        matrix_type = 'r-hrp'
    elif btn3 % 2:
        matrix_type = 'olo'
    elif btn4 % 2:
        matrix_type = 'r-olo'
    
    x, y, z = get_correlation(matrix_type)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=x,
        y=y,
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

def get_correlation(matrix_type):
    data_correlation, _ = get_data(name="correlation_matrix")
    processed_data, _ = get_data(name="processed_data")

    round_correlation = data_correlation.copy()
    threshold=1/(2*len(processed_data))
    for idx, raw in round_correlation.iterrows():
        for x in raw.index:
            to_round_value=raw[x]
            if -threshold<=to_round_value<=threshold:
                round_correlation.loc[idx, x]=0
            else:
                round_correlation.loc[idx, x] = round(to_round_value, 10)

    abs_correlation = round_correlation.copy() 
    for idx, raw in abs_correlation.iterrows():
        for x in raw.index:
            abs_correlation.loc[idx, x] = abs(raw[x])

    if matrix_type == 'hrp':
        distances = np.sqrt((1 - abs_correlation) / 2)
        res_order = compute_serial_matrix(distances.values, method='single')
        new_order = [abs_correlation.index[i] for i in res_order]

        ordered_correlation_matrix = round_correlation.reindex(index=new_order, columns=new_order) 
        z_value=ordered_correlation_matrix.to_numpy()
        x_value=[str(x)+"arg" for x in new_order]
        y_value=[str(x)+"arg" for x in new_order]
    elif matrix_type == 'r-hrp':
        all_new_order=innovative_correlation_clustering(round_correlation)
        new_test = round_correlation.reindex(index=all_new_order, columns=all_new_order)
        z_value=new_test.to_numpy()
        x_value=[str(x) + "arg" for x in new_test.columns]
        y_value=[str(x) + "arg" for x in new_test.index]

    elif matrix_type == 'olo':
        new_order=abs_optimal_leaf_ordering(data_correlation)
        ordered_correlation_matrix = round_correlation.reindex(index=new_order, columns=new_order)
        z_value = ordered_correlation_matrix.to_numpy()
        x_value = [str(x) + "arg" for x in new_order]
        y_value = [str(x) + "arg" for x in new_order]
    elif matrix_type == 'r-olo':
        all_new_order=my_optimal_leaf_ordering(round_correlation)
        new_test = round_correlation.reindex(index=all_new_order, columns=all_new_order)
        z_value = new_test.to_numpy()
        x_value = [str(x) + "arg" for x in new_test.columns]
        y_value = [str(x) + "arg" for x in new_test.index]
    else:
        z_value=round_correlation.to_numpy()
        x_value=[str(x)+"arg" for x in round_correlation.columns]
        y_value=[str(x)+"arg" for x in round_correlation.index]

    return x_value, y_value, z_value

@app.callback(
    [Output('feature_semantic_table', 'children'), Output('feature_cluster_table', 'children')],
    [Input("clustering-method", "value"),Input('confirm', 'submit_n_clicks'),Input("semantic-method-1", "value"),])
def get_feature_table(cluster_method, n_click, table_method):
    if len(os.listdir(PROCESSED_DIRECTORY)) == 12:
        group_table = pd.read_pickle(PROCESSED_DIRECTORY + "group_feature.pkl")
        cluster_table = pd.read_pickle(PROCESSED_DIRECTORY + table_method + "_" + cluster_method + "_cluster_feature.pkl")
    else:
        group_table, _  = get_data(n_click, "group_feature", False)
        cluster_table, _ = get_data(n_click, cluster_method + "_cluster_feature", False)
    
    print(group_table)
    print(cluster_table)
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