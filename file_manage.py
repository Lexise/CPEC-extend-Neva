import os
from os import listdir,unlink,walk
from os.path import isfile, join
import base64
from pathlib import Path
def clean_folder(folder_path):
    if len(listdir(folder_path))!=0:
        removed=[]
        for the_file in listdir(folder_path):
            file_path = join(folder_path, the_file)
            try:
                if isfile(file_path):
                    unlink(file_path)
                    removed.append(the_file)
            except Exception as e:
                print(e)
        return removed
    else:
        return []

def uploaded_files( directory ):
    """List the files in the upload directory."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    files = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def get_current_processed_dir_semantic(dir):
    files = uploaded_files(dir)
    semantics = []
    trailing='_processed_data.pkl'
    thelen= len(trailing)
    for x in files:
        if x[-thelen:] == trailing:
            semantic = x[:-thelen]
            semantics.append(semantic)
    return semantics


def save_file(name, content, dir):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    #content_type, content_string = content.split(',')
    with open(os.path.join(dir, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def clean_folder(folder_path):
    if len(listdir(folder_path))!=0:
        removed=[]
        for the_file in listdir(folder_path):
            file_path = join(folder_path, the_file)
            try:
                if isfile(file_path):
                    unlink(file_path)
                    removed.append(the_file)
            except Exception as e:
                print(e)
        return removed
    else:
        return []