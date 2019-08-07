import numpy as np 
import pandas as pd 
import json

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def squad_json_to_dataframe_train(input_file_path, record_path = ['data','paragraphs','qas','answers'],
                           verbose = 1):
    """
    """
    file = json.loads(open(input_file_path).read())

    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path )
    m = pd.io.json.json_normalize(file, record_path[:-1] )
    r = pd.io.json.json_normalize(file,record_path[:-2])
    
    idx = np.repeat(r['context'].values, r.qas.str.len())


    main['c_id'] = main['context'].factorize()[0]

#%%
input_file_path = './data/dev-v2.0.json'
file = json.loads(open(input_file_path).read())

#%%
record_path = ['data','paragraphs','qas','answers']

#%%
js = pd.io.json.json_normalize(file , record_path )
m = pd.io.json.json_normalize(file, record_path[:-1] )
r = pd.io.json.json_normalize(file, record_path[:-2] )


#%%
idx = np.repeat(r['context'].values, r.qas.str.len())
ndx = np.repeat(m['id'].values, m.answers.str.len())
m['context'] = idx
js['id'] = ndx

#%%
m.head()

#%%
js.head()

#%%
main = 
