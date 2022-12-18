

from azureml.core import Run
import pandas as pd
import os

run = Run.get_context()

data = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')

n_rows = data.shape[0]

run.log('# of rows:', n_rows)

os.makedirs('outputs', exist_ok=True)

data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

run.complete()