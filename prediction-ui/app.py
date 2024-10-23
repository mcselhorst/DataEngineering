import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask constructor
app = Flask(__name__)

@app.route('/checkheartdisease', methods=["GET", "POST"])
def check_heart_disease():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        prediction_input = [
            {
                "age": int(request.form.get("age")),
                "sex": int(request.form.get("sex")),
                "cp": int(request.form.get("cp")),
                "trestbps": int(request.form.get("trestbps")),
                "chol": int(request.form.get("chol")),
                "fbs": int(request.form.get("fbs")),
                "restecg": int(request.form.get("restecg")),
                "thalach": int(request.form.get("thalach")),
                "exang": int(request.form.get("exang")),
                "oldpeak": float(request.form.get("oldpeak")),
                "slope": int(request.form.get("slope")),
                "ca": int(request.form.get("ca")),
                "thal": int(request.form.get("thal"))
            }
        ]

        logging.debug("Prediction input : %s", prediction_input)

        predictor_api_url = os.environ['PREDICTOR_API']
        logging.debug("Predictor API URL: %s", predictor_api_url)

        try:
            res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))
            logging.debug("Response object: %s", res)
            logging.debug("Response text: %s", res.text)
            logging.debug("Response JSON: %s", res.json())

            prediction_value = res.json().get('result', 'No result found')
            logging.info("Prediction Output : %s", prediction_value)

            # Check for a test parameter to return JSON response
            if request.args.get('test') == 'true':
                return jsonify(prediction=prediction_value)

            return render_template("response_page.html",
                                   prediction_variable=prediction_value)
        except requests.exceptions.RequestException as e:
            logging.error("Error during prediction request: %s", e)
            return jsonify(message="Error during prediction request"), 500
        except ValueError as e:
            logging.error("Error parsing JSON response: %s", e)
            return jsonify(message="Error parsing JSON response"), 500

    else:
        return jsonify(message="Method Not Allowed"), 405

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)