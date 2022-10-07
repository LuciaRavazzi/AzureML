
from azureml.core import Run
import os

# get the experiment run the context.
run = Run.get_context()

# data = pd.read_csv("./data/countries-aggregated.csv")
data = {"A": [0,1,2,3], "B": [0,1,2,3], "C": [0,1,2,3]}

# log metric.
run.log_list('observations', data["A"])

run.complete()
