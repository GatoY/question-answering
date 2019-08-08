import numpy as np 
import pandas as pd 
import json

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


#%%
input_file_path = './data/train-v2.0.json'
file = json.loads(open(input_file_path).read())
record_path = ['data','paragraphs','qas','answers']
js = pd.io.json.json_normalize(file , record_path )
m = pd.io.json.json_normalize(file, record_path[:-1] )
r = pd.io.json.json_normalize(file, record_path[:-2] )
idx = np.repeat(r['context'].values, r.qas.str.len())
ndx = np.repeat(m['id'].values, m.answers.str.len())
m['context'] = idx
js['id'] = ndx
m = m.drop(['answers'], axis=1)
main = m.merge(js, on = 'id')
main.to_csv('data/raw_data.csv', index=False)
