import json
import os
import joblib  # Import joblib to load the scaler
import pandas as pd
from flask import jsonify
from keras.models import load_model # type: ignore
import logging
from io import StringIO

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None  # Add an attribute for the scaler

    def load_model_and_scaler(self):
        # Load the model and scaler if they are not already loaded
        if self.model is None:
            try:
                model_repo = os.environ['MODEL_REPO']
                model_path = os.path.join(model_repo, "heart_disease_model.h5")
                scaler_path = os.path.join(model_repo, "scaler.joblib")  # Path for the scaler
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)  # Load the scaler
                logging.info("Model and scaler loaded successfully.")
            except KeyError:
                logging.error("MODEL_REPO is undefined, loading from local.")
                self.model = load_model('heart_disease_model.h5')
                self.scaler = joblib.load('scaler.joblib')  # Load the scaler locally

    def predict_single_record(self, prediction_input):
        self.load_model_and_scaler()  # Ensure model and scaler are loaded
        logging.debug(prediction_input)

        # Convert input JSON to DataFrame
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')

        # Apply the StandardScaler to the input data
        df_scaled = self.scaler.transform(df)

        # Make predictions
        y_pred = self.model.predict(df_scaled)
        logging.info(y_pred[0])
        status = (y_pred[0] > 0.5)
        logging.info(type(status[0]))

        # Return the prediction outcome as a JSON message
        return jsonify({'result': str(status[0])}), 200
