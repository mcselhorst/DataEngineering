import unittest
import json
import os
from unittest.mock import patch, Mock
from app import app  # Import the Flask app

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set the environment variable for the predictor API URL
        os.environ['PREDICTOR_API'] = 'http://127.0.0.1:5001/heart_disease_predictor/' 

    @patch('requests.post')
    # Testing /checkheartdisease endpoint with mocking API response. So the behavior of the 
    # UI is tested without the API.

    def test_prediction_with_mocking(self, mock_post):
        # Create a mock response object with the desired properties
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'result': 'mocked_prediction'}
        mock_post.return_value = mock_response

        # Use the imported app
        with app.test_client() as client:
            response = client.post('/checkheartdisease?test=true', data={
                'age': "71",
                'sex': "0",
                'cp': "0",
                'trestbps': "112",
                'chol': "149",
                'fbs': "0",
                'restecg': "1",
                'thalach': "125",
                'exang': "0",
                'oldpeak': "1.6",
                'slope': "1",
                'ca': "0",
                'thal': "2"
            })

            # Print the response data for debugging
            print(response.data)

            # Check the response
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'mocked_prediction', response.data)

    def test_prediction_without_mocking(self):
        # Test UI with actual API and actual predictions
        with app.test_client() as client:
            response = client.post('/checkheartdisease?test=true', data={
                'age': "71",
                'sex': "0",
                'cp': "0",
                'trestbps': "112",
                'chol': "149",
                'fbs': "0",
                'restecg': "1",
                'thalach': "125",
                'exang': "0",
                'oldpeak': "1.6",
                'slope': "1",
                'ca': "0",
                'thal': "2"
            })

            # Print the response data for debugging
            print(response.data)

            # Check the response
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'prediction', response.data)

if __name__ == '__main__':
    unittest.main()