# DataEngineering

### Instructions to Replicate Current UI + API heart disease checker in Google Cloud:

1. **Connect with your repository**.
   - Ensure you are connected with your repository.

2. **Docker image in Artifact Registry**.
   - The Docker image for this project should be in Artifact Registry (tutorial available in labs).

3. **Create a trigger in Cloud Build**.
   - Set up a manual trigger using the following build configuration from the repository:
     ```plaintext
     pipelines/cloud_build_ml_app.json
     ```

4. **Retrieve the prediction API URL from Cloud Run**.
   - After the build is finished, go to Cloud Run and copy the `prediction-api` URL.

5. **Update the environment variable in prediction-ui**.
   - Go to `prediction-ui` and click **Edit & Deploy New Revision**.
   - Paste the `prediction-api` URL into the `Environment variable` field.
   - Ensure the environment variable value for `PREDICTOR_API` looks like this:
     ```plaintext
     https://prediction-api-905553022046.us-central1.run.app/heart_disease_predictor/
     ```

6. **Access the prediction UI**.
   - Click on the `prediction-ui` URL, and add `checkheartdisease` at the end, like this:
     ```plaintext
     https://prediction-ui-905553022046.us-central1.run.app/checkheartdisease
     ```

Now, you should be able to test the application.
