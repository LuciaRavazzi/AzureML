import json
import joblib
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'my_diabetes_model.pkl' )
    model = joblib.load(model_path)

def run(raw_data):

    data = np.array(json.loads(raw_data)['data'])
    predictions = model.predict(data)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)